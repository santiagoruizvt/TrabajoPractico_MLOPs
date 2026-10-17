import os
import io
import logging
import zipfile
from datetime import datetime, timezone
 
import boto3
import requests
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
 
logger = logging.getLogger(__name__)
 
UCDP_DOWNLOAD_URL = "https://ucdp.uu.se/downloads/ged/ged241-csv.zip"
S3_BUCKET         = "data"
S3_KEY            = "ucdp_ged.csv"
 
 
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
    )
 
 
@dag(
    dag_id="ucdp_download_dataset",
    description=(
        "Descarga el dataset UCDP-GED y lo sube a MinIO. "
        "Idempotente: si el archivo ya existe en s3://data/, no hace nada."
    ),
    schedule="@once",
    start_date=datetime(2026, 9, 26, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    tags=["ucdp", "dataset", "init"],
)
def download_dataset_dag():
 
    @task()
    def check_if_exists() -> bool:
        """
        Verifica si el CSV ya existe en MinIO.
        Retorna True si existe (nada que hacer), False si hay que descargarlo.
        """
        s3 = get_s3_client()
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
            logger.info(f"✅ s3://{S3_BUCKET}/{S3_KEY} ya existe. Nada que hacer.")
            return True
        except Exception:
            logger.info(f"⬇️  s3://{S3_BUCKET}/{S3_KEY} no encontrado. Descargando...")
            return False
 
    @task()
    def download_extract_and_upload(already_exists: bool) -> None:
        """
        Descarga el ZIP, lo descomprime en memoria y sube el CSV a MinIO
        todo en una sola tarea para evitar problemas de /tmp entre procesos.
        """
        if already_exists:
            logger.info("Archivo ya existe en MinIO. Tarea completada sin acción.")
            return
 
        s3 = get_s3_client()
 
        # ── Crear bucket si no existe ─────────────────────────────
        try:
            s3.head_bucket(Bucket=S3_BUCKET)
            logger.info(f"Bucket '{S3_BUCKET}' ya existe.")
        except Exception:
            logger.info(f"Creando bucket '{S3_BUCKET}'...")
            s3.create_bucket(Bucket=S3_BUCKET)
 
        # ── Descargar ZIP en memoria ──────────────────────────────
        logger.info(f"Descargando desde {UCDP_DOWNLOAD_URL} ...")
        response = requests.get(UCDP_DOWNLOAD_URL, stream=True, timeout=600)
        response.raise_for_status()
 
        zip_buffer = io.BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            zip_buffer.write(chunk)
            downloaded += len(chunk)
 
        size_mb = downloaded / (1024 * 1024)
        logger.info(f"ZIP descargado en memoria: {size_mb:.1f} MB")
 
        # ── Extraer CSV del ZIP en memoria ────────────────────────
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError("No se encontró ningún CSV dentro del ZIP.")
 
            csv_name = csv_files[0]
            logger.info(f"Extrayendo '{csv_name}' del ZIP...")
 
            csv_bytes = zf.read(csv_name)
 
        csv_size_mb = len(csv_bytes) / (1024 * 1024)
        logger.info(f"CSV extraído en memoria: {csv_size_mb:.1f} MB")
 
        # ── Subir CSV a MinIO directamente desde memoria ──────────
        logger.info(f"Subiendo a s3://{S3_BUCKET}/{S3_KEY} ...")
        s3.upload_fileobj(
            Fileobj=io.BytesIO(csv_bytes),
            Bucket=S3_BUCKET,
            Key=S3_KEY,
        )
        logger.info(f"✅ Dataset disponible en s3://{S3_BUCKET}/{S3_KEY}")
 
    # ── Flujo ─────────────────────────────────────────────────────
    exists = check_if_exists()
    uploaded =download_extract_and_upload(exists)

    trigger_model = TriggerDagRunOperator(
        task_id="trigger_model_lifecycle",
        trigger_dag_id="ucdp_model_lifecycle",
        wait_for_completion=False,
    )
 
    uploaded >> trigger_model
 
 
download_dataset_dag()