## Changelog desde la entrega parcial

- Se agrega un DAG `ucdp_download_dataset` que verifica si el ucdp-ged.csv se encuentra cargado en minIO. Si el mismo no se encuentra, lo descarga.
- Se modifica el `docker-compose.yaml`, eliminando el `init-mlflow` y agregando la api como profile `all`
- Se actualizó `README.md` con descripción del proyecto, integrantes del grupo, tabla de arquitectura, sección de verificación en MLflow y ejemplos de requests a la API.
- Se actualizó `dockerfiles/fastapi/app.py` agregando los endpoints `GET /model/info` y `GET /predict/example`, y se incorporó `model_version` en la respuesta de `GET /health`.
- Se resolvieron conflictos de merge con upstream incorporando la imagen `coollabsio/minio` y actualizando dependencias de MLflow