from datetime import datetime
import os
import sys

from airflow.decorators import dag, task

MODEL_NAME = "ucdp-severity-rf"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@dag(
    dag_id="ucdp_model_lifecycle",
    description="Train, register, and promote the UCDP-GED severity model.",
    schedule=None,
    start_date=datetime(2026, 9, 26),
    catchup=False,
    max_active_runs=1,
    tags=["ucdp", "mlflow"],
)
def ucdp_model_lifecycle():
    @task(retries=0)
    def train_model() -> str:
        sys.path.insert(0, "/opt/mlops/scripts")
        from init_mlflow_experiment import train_and_log_model

        return train_and_log_model()

    @task(retries=2)
    def register_model(mlflow_run_id: str) -> str:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)

        # Reuse a version if this task is retried after registration succeeded.
        existing = next(
            (
                version
                for version in client.search_model_versions(
                    f"name='{MODEL_NAME}'"
                )
                if version.run_id == mlflow_run_id
            ),
            None,
        )
        if existing is not None:
            return str(existing.version)

        model_version = mlflow.register_model(
            model_uri=f"runs:/{mlflow_run_id}/model",
            name=MODEL_NAME,
        )
        return str(model_version.version)

    @task(retries=2)
    def promote_model(version: str) -> str:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_URI)
        version = str(version)
        target = client.get_model_version(MODEL_NAME, version)

        if target.current_stage != "Production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=False,
            )

        # Promote first so a failure while archiving does not leave no Production model.
        for previous in client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        ):
            if str(previous.version) != version:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=previous.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        return version

    run_id = train_model()
    version = register_model(run_id)
    promote_model(version)


ucdp_model_lifecycle()
