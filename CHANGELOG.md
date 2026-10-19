## Changelog

### Entrega parcial
- Se agrega un DAG `ucdp_download_dataset` que verifica si el ucdp-ged.csv se encuentra cargado en MinIO. Si el mismo no se encuentra, lo descarga.
- Se modifica el `docker-compose.yaml`, eliminando el `init-mlflow` y agregando la API como profile `all`.
- Se actualizó `README.md` con descripción del proyecto, integrantes del grupo, tabla de arquitectura, sección de verificación en MLflow y ejemplos de requests a la API.
- Se actualizó `dockerfiles/fastapi/app.py` agregando los endpoints `GET /model/info` y `GET /predict/example`, y se incorporó `model_version` en la respuesta de `GET /health`.
- Se resolvieron conflictos de merge con upstream incorporando la imagen `coollabsio/minio` y actualizando dependencias de MLflow a 3.x.

### Entrega final
- El preprocesamiento pasa **dentro del Pipeline** (`scripts/ucdp_features.py`). El modelo registrado en MLflow recibe el evento crudo; el módulo viaja con el artefacto.
- `POST /predict` y `/predict/batch` reciben ahora el evento crudo en vez de 22 features precalculadas.
- `ucdp_model_lifecycle` es el único camino de entrenamiento. Se elimina `scripts/init_mlflow_experiment.py` y el perfil `manual-init`.
- Nuevo endpoint `POST /model/reload`; el DAG llama al endpoint al promover el modelo, evitando reiniciar la API.
- `mlflow==3.16.1` fijado en todas las imágenes.
- README reescrito: problema, dataset, integrantes, arquitectura (diagrama), orden de ejecución, verificación en MLflow y ejemplos de requests. Se eliminó el texto de la cátedra que ya no aplicaba.
- Se agregan screenshots de validación del funcionamiento de todos los servicios que integran la arquitectura del modelo.