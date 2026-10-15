from datetime import datetime
import os
import sys

from airflow.decorators import dag, task

MODEL_NAME = "ucdp-severity-rf"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8800")


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
        # get_latest_versions only returns the highest version per stage, so it
        # would never see the previous Production version; list them all instead.
        for previous in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if (
                previous.current_stage == "Production"
                and str(previous.version) != version
            ):
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=previous.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        return version

    @task(retries=2)
    def reload_api_model(version: str) -> None:
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"{FASTAPI_URL}/model/reload", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError:
            # The API is up but failed to load the model: retry / fail the task.
            raise
        except urllib.error.URLError as exc:
            # The API is not running; it will load the Production model on startup.
            print(f"FastAPI not reachable at {FASTAPI_URL} ({exc.reason}); skipping reload.")
            return

        print(f"FastAPI reload: {body}")
        if body.get("current_version") != str(version):
            raise RuntimeError(
                f"FastAPI is serving version {body.get('current_version')}, "
                f"expected {version}."
            )

    run_id = train_model()
    version = register_model(run_id)
    promoted = promote_model(version)
    reload_api_model(promoted)


ucdp_model_lifecycle()
