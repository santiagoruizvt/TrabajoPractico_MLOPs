# Changelog

Cambios desde la entrega parcial.

## Entrega final

### Modelo y servicio
- El preprocesamiento (target encoding del país, frecuencia de díada, one-hot, features temporales) pasa **dentro del Pipeline** (`scripts/ucdp_features.py`, transformer `UCDPFeatureBuilder`). El modelo registrado en MLflow recibe el evento crudo del UCDP-GED; el módulo viaja con el artefacto (`code_paths`).
- `POST /predict` y `/predict/batch` reciben ahora el evento crudo (`country_id`, `dyad_new_id`, `date_start`, `region`, ...) en vez de 22 features precalculadas. Se actualizan `GET /predict/example`, `/model/info` y `scripts_aux/manual_tests.ps1`.
- Las estadísticas de `dyad_freq` y `country_te` se aprenden solo con train; en test/serving una díada no vista se imputa con la mediana. Las métricas pueden variar levemente respecto de la entrega parcial.
- `mlflow==3.16.1` fijado en la imagen de FastAPI, igual que en Airflow.

### Un solo camino de entrenamiento
- El DAG `ucdp_model_lifecycle` es el único camino. `scripts/init_mlflow_experiment.py` pasa a `scripts/ucdp_training.py` (sin `main()`); se eliminan `scripts/Dockerfile.init` y `scripts/requirements.txt`. Se quita `init-mlflow` del compose y el perfil `manual-init` de la API (queda en `all`).

### Dato y orquestación
- Nuevo DAG `ucdp_download_dataset`: descarga el UCDP-GED a `s3://data/ucdp_ged.csv` si no existe y dispara `ucdp_model_lifecycle`. Corre solo al levantar Airflow (`@once`); `start_date` fijo en vez de `datetime.now()`.
- `ucdp_model_lifecycle` archiva la versión que estaba en Production al promover la nueva y llama a `POST /model/reload`, de modo que la API sirve el modelo nuevo sin reiniciar.
- Nuevo endpoint `POST /model/reload`; `GET /health` informa `model_version`; nuevos `GET /model/info` y `GET /predict/example`.

### Documentación
- README reescrito: problema, dataset, integrantes, arquitectura (diagrama), orden de ejecución, verificación en MLflow y ejemplos de requests. Se eliminó el texto de la cátedra que ya no aplicaba.
