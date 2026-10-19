# Clasificador de Severidad de Eventos de Violencia Organizada — UCDP-GED

**MLOps1 — CEIA — FIUBA · Grupo 8**
- Santiago Ruiz
- Maximiliano Antonio Cencic
- Jeraldinne Del Valle Molleda Guanipa

---

## Qué hace y para qué

Dado un evento de violencia organizada del [UCDP Georeferenced Event Dataset (GED) v24.1](https://ucdp.uu.se/downloads/) (país, región, díada de actores, fecha, tipo de violencia, precisión geográfica/temporal), el servicio predice su **severidad**: `Bajo`, `Medio` o `Alto`.

- **Problema:** clasificación multiclase supervisada. El target se obtiene discretizando las muertes estimadas (`best`) con los percentiles 33 y 66.
- **Modelo:** Random Forest (hiperparámetros del trabajo de Aprendizaje de Máquina 1), Macro F1 ≈ 0.57 en test.
- **Qué aporta MLOps:** el ciclo completo (dato → entrenamiento → registro → promoción → servicio) corre de punta a punta y sin pasos manuales, sobre servicios en Docker Compose.

El modelo registrado es un **Pipeline sklearn único** que recibe el evento **crudo** (columnas del UCDP-GED) y aplica internamente target encoding del país, frecuencia de díada, one-hot, features temporales y escalado. La API no precalcula nada.

## Arquitectura

```mermaid
flowchart LR
    UCDP[ucdp.uu.se] -->|descarga| D1
    subgraph Airflow
        D1[DAG ucdp_download_dataset] -->|dispara| D2[DAG ucdp_model_lifecycle<br/>train → register → promote → reload]
    end
    D1 -->|sube CSV| S3[(MinIO<br/>s3://data)]
    D2 -->|lee CSV| S3
    D2 -->|run, métricas, modelo| ML[MLflow<br/>+ Postgres]
    ML -->|artefactos| S3b[(MinIO<br/>s3://mlflow)]
    D2 -->|POST /model/reload| API[FastAPI]
    API -->|carga modelo en Production| ML
    Cliente -->|/predict, /predict/batch| API
```

| Servicio | Rol | URL |
|---|---|---|
| Apache Airflow | Orquesta descarga del dato y ciclo de vida del modelo | http://localhost:8080 (airflow / airflow) |
| MLflow | Tracking de experimentos y Model Registry | http://localhost:5001 |
| MinIO | Data lake S3: `s3://data` (dato crudo) y `s3://mlflow` (artefactos) | http://localhost:9001 (minio / minio123) |
| FastAPI | Sirve el modelo en Production | http://localhost:8800 · docs en `/docs` |
| PostgreSQL / Valkey | Metadatos de Airflow y MLflow / broker de Celery | internos |

Estructura relevante:

```
airflow/dags/download_dataset.py        DAG: descarga UCDP-GED -> s3://data/ucdp_ged.csv
airflow/dags/ucdp_model_lifecycle.py    DAG: train -> register -> promote -> reload API
scripts/ucdp_training.py                Entrenamiento (lo invoca el DAG; único camino)
scripts/ucdp_features.py                Feature engineering como transformer sklearn
dockerfiles/fastapi/app.py              API
notebook_UCDP/                          Notebook de análisis y selección del modelo
```

## Cómo reproducirlo

Requisitos: Docker y Docker Compose. En Linux/macOS ajustar `AIRFLOW_UID` en `.env` con tu `id -u`.

1. Levantar todo (incluye la API, que arranca sin modelo y lo toma cuando existe):

   ```bash
   docker compose --profile all up --build
   ```

2. Esperar a que los servicios estén *healthy* (`docker ps`).

3. **No hay que hacer nada más.** `ucdp_download_dataset` corre solo al levantar Airflow (`@once`, sin pausa inicial):
   1. Descarga el UCDP-GED si `s3://data/ucdp_ged.csv` no existe (si ya está, lo saltea).
   2. Dispara `ucdp_model_lifecycle`: entrena, registra `ucdp-severity-rf`, lo promueve a `Production` (archivando la versión anterior) y llama a `POST /model/reload` para que la API sirva la nueva versión sin reiniciar.

   Para reentrenar: disparar `ucdp_model_lifecycle` (o `ucdp_download_dataset`) desde la UI de Airflow.

   > Si hay una versión en Production, la API la carga al arrancar. Antes de que el DAG termine, `/predict` responde `503` y `/health` responde `degraded`.

4. Verificar (ver abajo) y probar la API.

Sin acceso a internet, subir el CSV a mano como `s3://data/ucdp_ged.csv` desde la consola de MinIO; el DAG detecta que existe y no lo descarga.

### Verificar en MLflow

1. Abrir http://localhost:5001 → experimento `ucdp-severity-classification`.
2. La run `rf_optimizado_notebook_params` tiene las métricas `train_f1_macro`, `test_f1_macro`, `train_accuracy`, `test_accuracy`, `overfit_gap_f1`.
3. **Models → `ucdp-severity-rf`**: la versión más reciente está en `Production` y las anteriores en `Archived`.
4. `curl http://localhost:8800/health` devuelve `"status": "ok"` y la misma versión.

## Ejemplos de requests

La API recibe el evento crudo. `best` y `deaths_civilians` son opcionales (si faltan, `civilian_ratio` = 0).

```bash
curl http://localhost:8800/health
curl http://localhost:8800/model/info
curl http://localhost:8800/predict/example      # payload de ejemplo

curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{
    "country_id": 652,
    "dyad_new_id": 123,
    "date_start": "2015-03-14",
    "latitude": 33.5,
    "longitude": 36.3,
    "region": "Middle East",
    "type_of_violence": 1,
    "where_prec": 1,
    "date_prec": 1,
    "event_clarity": 1,
    "best": 10,
    "deaths_civilians": 5
  }'
# Respuesta: {"severity_code": ..., "severity_label": ..., "probabilities": {"Bajo": ..., "Medio": ..., "Alto": ...}}

# Recarga manual del modelo en Production (el DAG ya lo hace solo)
curl -X POST http://localhost:8800/model/reload
```

Desde Python:

```python
import requests
base = "http://localhost:8800"
payload = requests.get(f"{base}/predict/example").json()
print(requests.post(f"{base}/predict", json=payload).json())
print(requests.post(f"{base}/predict/batch", json=[payload, payload]).json())  # máx. 100
```

Valores válidos: `region` ∈ {Africa, Americas, Asia, Europe, Middle East}; `type_of_violence` ∈ {1, 2, 3}; `where_prec` 1–7; `date_prec` 1–5; `event_clarity` 1–2. Swagger interactivo en http://localhost:8800/docs.

## Apagar los servicios

```bash
docker compose --profile all down                  # detiene
docker compose down --rmi all --volumes            # además borra imágenes, buckets y bases
```

## Configuración

Puertos y credenciales se cambian en `.env`. Las conexiones y variables de Airflow están en `airflow/secrets/`. Para usar MinIO desde tu máquina (boto3, awscli):

```bash
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
AWS_ENDPOINT_URL_S3=http://localhost:9000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

---
 
## Validación de los servicios
 
Una vez que todos los servicios estén funcionando (verificá con `docker ps -a` que estén *healthy*, o revisá en Docker Desktop), podés acceder a los diferentes servicios mediante:
 
- **Apache Airflow:** http://localhost:8080
- **MLflow:** http://localhost:5001
- **MinIO:** http://localhost:9001 (ventana de administración de buckets)
- **API:** http://localhost:8800
- **Documentación de la API:** http://localhost:8800/docs
A continuación se muestran capturas de cada servicio funcionando con este proyecto, que evidencian el ciclo completo: dato → entrenamiento → registro → promoción → servicio.
 
### 1. Apache Airflow — orquestación
 
Los dos DAGs del proyecto están cargados y activos: `ucdp_download_dataset` (programado `@once`, ejecutado con éxito) y `ucdp_model_lifecycle` (tags `ucdp`, `mlflow`).
 
![Airflow - listado de DAGs](screenshots/airflow_dags.png)
 
`ucdp_download_dataset` dispara automáticamente `ucdp_model_lifecycle`, que aparece en ejecución (run `manual`, disparado por el DAG anterior):
 
![Airflow - run del DAG ucdp_download_dataset y disparo de ucdp_model_lifecycle](screenshots/aiflow_DAGs_run_ucdp_download_dataset.png)
 
Detalle de las tareas de `ucdp_model_lifecycle`, todas en `success`: `train_model` → `register_model` → `promote_model` → `reload_api_model`.
 
![Airflow - tareas del DAG ucdp_model_lifecycle](screenshots/tasks_model_airflow.png)
 
### 2. MinIO — data lake
 
El bucket `s3://data` contiene el dataset crudo `ucdp_ged.csv`, subido por el DAG de descarga.
 
![MinIO - bucket data con ucdp_ged.csv](screenshots/miniio.png)
 
### 3. MLflow — tracking y Model Registry
 
Experimento `ucdp-severity-classification`. Cada ejecución del DAG genera una run (fuente: `airflow`) y registra una nueva versión del modelo `ucdp-severity-rf` (v1, v2, v3, v4, ...):
 
![MLflow - training runs](screenshots/training_runs_mlflow.png)
 
Vista complementaria del experimento con el panel del MLflow Assistant:
 
![MLflow - experimento y runs](screenshots/mlflow.png)
 
### 4. FastAPI — servicio del modelo
 
Swagger UI (`/docs`) con todos los endpoints documentados: `/`, `/health`, `/model/info`, `/model/reload`, `/predict/example`, `/predict` y `/predict/batch`.
 
![FastAPI - documentación Swagger](screenshots/fastapi_docs.png)
 
### 5. Prueba de los endpoints
 
Prueba de los endpoints desde la terminal. Se verifica que la API tiene cargado `ucdp-severity-rf` en stage `Production`, con sus métricas y parámetros (`/health`, `/model/info`):
 
![Test de endpoints - health y model/info](screenshots/testendpoints-2.png)
 
![Test de endpoints - reload, predict y predict/batch](screenshots/testendpoints-1.png)

Base del entorno: [amq2-service-ml](https://github.com/facundolucianna/amq2-service-ml) (CEIA, FIUBA), adaptado para este trabajo.
