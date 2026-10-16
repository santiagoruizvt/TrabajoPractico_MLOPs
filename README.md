# Clasificador de Severidad de Eventos de Violencia Organizada — UCDP-GED

### MLOps1 - CEIA - FIUBA

**Grupo 8**
- Santiago Ruiz
- Maximiliano Antonio Cencic
- Jeraldinne Del Valle Molleda Guanipa

---

## Contexto

Este repositorio implementa en un entorno productivo el modelo de Machine Learning desarrollado en la materia Aprendizaje de Máquina 1. El modelo predice la **severidad** (Bajo / Medio / Alto) de eventos de violencia organizada a partir del dataset [UCDP Georeferenced Event Dataset (GED) v24.1](https://ucdp.uu.se/downloads/).

El problema es de **clasificación multiclase supervisada**: dado un evento de violencia con sus características contextuales (país, región, actores, fecha, precisión geográfica), el modelo predice cuán letal fue ese evento. El modelo final es un **Random Forest** optimizado con Macro F1 = 0.57 en test.

### MLOps1 - CEIA - FIUBA
Estructura de servicios para la implementación del proyecto final de MLOps1 - CEIA - FIUBA

Supongamos que trabajamos para **ML Models and something more Inc.**, la cual ofrece un servicio que proporciona modelos mediante una REST API. Internamente, tanto para realizar tareas de DataOps como de MLOps, la empresa cuenta con varios servicios que ayudan a ejecutar las acciones necesarias. También dispone de un Data Lake en S3, para este caso, simularemos un S3 utilizando MinIO.

Para simular esta empresa, utilizaremos Docker y, a través de Docker Compose, desplegaremos varios contenedores que representan distintos servicios en un entorno productivo.

Los servicios que contamos son:
- [Apache Airflow](https://airflow.apache.org/)
- [MLflow](https://mlflow.org/)
- API Rest para servir modelos ([FastAPI](https://fastapi.tiangolo.com/))
- [MinIO](https://min.io/)
- Base de datos relacional [PostgreSQL](https://www.postgresql.org/)
- Base de dato key-value [ValKey](https://valkey.io/) 

![Diagrama de servicios](final_assign.png)

Por defecto, cuando se inician los multi-contenedores, se crean los siguientes buckets:

- `s3://data`
- `s3://mlflow` (usada por MLflow para guardar los artefactos).

y las siguientes bases de datos:

- `mlflow_db` (usada por MLflow).
- `airflow` (usada por Airflow).

## Tarea a realizar

La tarea es implementar el modelo que desarrollaron en Aprendizaje de Máquina en este ambiente productivo. Para ello, pueden usar y crear los buckets y bases de datos que necesiten. Lo mínimo que deben realizar es:

- Un DAG en Apache Airflow. Puede ser cualquier tarea que se desee realizar, como entrenar el modelo, un proceso ETL, etc.
- Un experimento en MLflow de búsqueda de hiperparámetros.
- Servir el modelo implementado en AMq1 en el servicio de RESTAPI.
- Documentar (comentarios y docstring en scripts, notebooks, y asegurar que la documentación de FastAPI esté de acuerdo al modelo).

Desde **ML Models and something more Inc.** autorizan a extender los requisitos mínimos. También pueden utilizar nuevos servicios (por ejemplo, una base de datos no relacional, otro orquestador como MetaFlow, un servicio de API mediante NodeJs, etc.).

### Ejemplo 

El [branch `example_implementation`](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation) contiene un ejemplo de aplicación para guiarse. Se trata de una implementación de un modelo de clasificación utilizando los datos de [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

Además se cuenta con una implementación ejemplo de predicción en bache con una parte que funciona gran parte de local en [branch `batch-example`](https://github.com/facundolucianna/amq2-service-ml/tree/example_implementation)

## Instalación

1. Para poder levantar todos los servicios, primero instala [Docker](https://docs.docker.com/engine/install/) en tu computadora (o en el servidor que desees usar).
2. Clona este repositorio.
3. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando `id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en `airflow/dags`) o plugins, etc.
4. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

5. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` que todos los servicios estén healthy o revisa en Docker Desktop), podrás acceder a los diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

7. Para inicializar el modelo, ingresá a Apache Airflow (http://localhost:8080) y triggeareá manualmente el DAG `ucdp_download_dataset`. Este DAG encadena automáticamente todo el flujo de inicialización:

   - **Si el dataset no está en MinIO:** lo descarga desde [ucdp.uu.se](https://ucdp.uu.se/downloads/), lo descomprime en memoria y lo sube a `s3://data/ucdp_ged.csv`. Si el dataset ya existe, este paso se saltea.
   - **Una vez disponible el dataset:** se triggerea automáticamente el DAG `ucdp_model_lifecycle`, que realiza la búsqueda de hiperparámetros, entrena el modelo final, lo registra en MLflow y lo promueve al stage `Production`.

   Una vez que `ucdp_model_lifecycle` finaliza con éxito, la API está lista para recibir predicciones. Podés verificarlo en http://localhost:8800/health, que indicará el nombre y versión del modelo en Production.

   > **Nota:** la API levanta junto con el resto de los servicios en el paso 6. Si intentás hacer una predicción antes de que el DAG termine, el endpoint `/predict` devolverá un error `503` indicando que el modelo aún no está disponible. El endpoint `/health` siempre responde independientemente del estado del modelo.

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP (puede ser una privada si tu servidor está en tu LAN o una IP pública si no; revisa firewalls u otras reglas que eviten las conexiones).

Todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`. Se invita a jugar y romper para aprender; siempre puedes volver a clonar este repositorio.

---

## Verificar en MLflow

Una vez que el DAG `ucdp_model_lifecycle` termina, podés verificar en MLflow:

1. Abrí http://localhost:5001
2. Entrá al experimento `ucdp-severity-classification`
3. Verificá que existe una run con las métricas `f1_test`, `f1_train` y `acc_test`
4. En **Models** → `ucdp-severity-rf` → verificá que la versión más reciente está en stage `Production`

---

## Testear la API

### Desde la terminal con curl

```bash
# Verificar estado del servicio y del modelo
curl http://localhost:8800/health

# Ver información del modelo cargado (versión, features, métricas)
curl http://localhost:8800/model/info

# Obtener un ejemplo de payload listo para usar
curl http://localhost:8800/predict/example

# Realizar una predicción
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{
    "civilian_ratio": 0.5,
    "country_te": 1.2,
    "date_prec": 1,
    "dyad_freq": 100.0,
    "dyad_id": 123,
    "era_1989-99": 0,
    "era_2000-09": 0,
    "event_clarity": 1,
    "latitude": 33.5,
    "longitude": 36.3,
    "month_cos": 0.5,
    "month_sin": 0.8,
    "region_Africa": 0,
    "region_Americas": 0,
    "region_Asia": 0,
    "region_Europe": 0,
    "region_Middle East": 1,
    "tov_1": 1,
    "tov_2": 0,
    "tov_3": 0,
    "where_prec": 1,
    "year": 2015
  }'
```

### Desde Python

```python
import requests

# Health check
print(requests.get("http://localhost:8800/health").json())

# Obtener ejemplo y predecir
payload = requests.get("http://localhost:8800/predict/example").json()
print(requests.post("http://localhost:8800/predict", json=payload).json())
```

### Desde el navegador

Accedé a http://localhost:8800/docs para el Swagger UI interactivo. Cada endpoint tiene documentación, descripción de campos y un botón **Try it out** para ejecutar requests en vivo.

---

## Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no se están utilizando, se recomienda detenerlos. Para hacerlo, ejecuta el siguiente comando:

```bash
docker compose --profile all down
```

Si deseas no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), utiliza el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Si haces esto, perderás todo en los buckets y bases de datos.

## Aspectos específicos de Airflow

### Variables de entorno
Airflow ofrece una amplia gama de opciones de configuración. En el archivo `docker-compose.yaml`, dentro de `x-airflow-common`, se encuentran variables de entorno que pueden modificarse para ajustar la configuración de Airflow. Pueden añadirse [otras variables](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html).

### Uso de ejecutores externos
Actualmente, para este caso, Airflow utiliza un ejecutor [celery](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html), lo que significa que las tareas se ejecutan en otro contenedor. 

### Uso de la CLI de Airflow

Si necesitan depurar Apache Airflow, pueden utilizar la CLI de Apache Airflow de la siguiente manera:

```bash
docker compose --profile all --profile debug up
```

Una vez que el contenedor esté en funcionamiento, pueden utilizar la CLI de Airflow de la siguiente manera, 
por ejemplo, para ver la configuración:

```bash
docker-compose run airflow-cli config list      
```

Para obtener más información sobre el comando, pueden consultar [aqui](https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html).

### Variables y Conexiones

Si desean agregar variables para accederlas en los DAGs, pueden hacerlo en `secrets/variables.yaml`. Para obtener más [información](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html), 
consulten la documentación.

Si desean agregar conexiones en Airflow, pueden hacerlo en `secrets/connections.yaml`. También es posible agregarlas mediante la interfaz de usuario (UI), pero estas no persistirán si se borra todo. Por otro lado, cualquier conexión guardada en `secrets/connections.yaml` no aparecerá en la UI, aunque eso no significa que no exista. Consulten la documentación para obtener más 
[información](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html).

## Conexión con los buckets

Dado que no estamos utilizando Amazon S3, sino una implementación local de los mismos mediante MinIO, es necesario modificar las variables de entorno para conectar con el servicio de MinIO. Las variables de entorno son las siguientes:

```bash
AWS_ACCESS_KEY_ID=minio   
AWS_SECRET_ACCESS_KEY=minio123 
AWS_ENDPOINT_URL_S3=http://localhost:90000
```

MLflow también tiene una variable de entorno que afecta su conexión a los buckets:

```bash
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```
Asegúrate de establecer estas variables de entorno antes de ejecutar tu notebook o scripts en tu máquina o en cualquier otro lugar. Si estás utilizando un servidor externo a tu computadora de trabajo, reemplaza localhost por su dirección IP.

Al hacer esto, podrás utilizar `boto3`, `awswrangler`, etc., en Python con estos buckets, o `awscli` en la consola.

Si tienes acceso a AWS S3, ten mucho cuidado de no reemplazar tus credenciales de AWS. Si usas las variables de entorno, no tendrás problemas.

## Valkey

La base de datos Valkey es usada por Apache Airflow para su funcionamiento. Tal como está configurado ahora no esta expuesto el puerto para poder ser usado externamente. Se puede modificar el archivo `docker-compose.yaml` para habilitaro.

## Pull Request

Este repositorio está abierto para que realicen sus propios Pull Requests y así contribuir a mejorarlo. Si desean realizar alguna modificación, **¡son bienvenidos!** También se pueden crear nuevos entornos productivos para aumentar la variedad de implementaciones, idealmente en diferentes `branches`. Algunas ideas que se me ocurren que podrían implementar son:

- Reemplazar Airflow y MLflow con [Metaflow](https://metaflow.org/) o [Kubeflow](https://www.kubeflow.org).
- Reemplazar MLflow con [Seldon-Core](https://github.com/SeldonIO/seldon-core).
- Agregar un servicio de tableros como, por ejemplo, [Grafana](https://grafana.com).