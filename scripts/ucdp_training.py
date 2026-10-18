"""
ucdp_training.py
================
Entrenamiento del modelo de severidad UCDP-GED. Lo invoca el DAG
`ucdp_model_lifecycle` (tarea train_model); es el único camino de entrenamiento.

  1. Lee el dataset UCDP-GED desde MinIO (s3://data/ucdp_ged.csv)
  2. Entrena un Pipeline sklearn que incluye TODO el preprocesamiento
     (ver ucdp_features.py), de modo que recibe el evento crudo.
  3. Registra parámetros, métricas y el pipeline como artefacto de una run en
     MLflow. El registro y la promoción a Production los hace el DAG.
"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import skops.io

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from ucdp_features import UCDPFeatureBuilder, RAW_COLUMNS, SELECTED_FEATS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. Configuración
# ─────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI",    "http://localhost:5000")
DATA_PATH       = os.getenv("DATA_BUCKET",             "s3://data/ucdp_ged.csv")
EXPERIMENT_NAME = "ucdp-severity-classification"

# Estos son los mejores hiperparámetros encontrados en el notebook
# (RandomizedSearchCV, 20 iteraciones, 5 folds, scoring=f1_macro)
BEST_RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         30,
    "min_samples_split": 50,
    "min_samples_leaf":  1,
    "max_features":      "log2",
    "class_weight":      "balanced",
    "random_state":      42,
    "n_jobs":            -1,
}

# Columnas que usan RobustScaler (sesgadas, del notebook celda 56)
SKEWED_FEATS   = ["dyad_freq", "country_te"]
# Columnas que usan StandardScaler (continuas simétricas, celda 56)
STANDARD_FEATS = [
    "civilian_ratio", "latitude", "longitude",
    "month_sin", "month_cos", "year",
    "where_prec", "event_clarity", "date_prec",
]
# Columnas binarias / one-hot: no se escalan
PASSTHROUGH_FEATS = [f for f in SELECTED_FEATS
                     if f not in SKEWED_FEATS + STANDARD_FEATS]


# ─────────────────────────────────────────────────────────────────
# 1. Carga del dataset desde MinIO / ruta local
# ─────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Lee el CSV desde MinIO (s3://) o desde ruta local.
    MinIO se accede como S3 gracias a las variables de entorno
    MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    que boto3 lee automáticamente.
    """
    print(f"[1/5] Cargando dataset desde {path} ...")
    if path.startswith("s3://"):
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )
        bucket, key = path[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df  = pd.read_csv(obj["Body"])
    else:
        df = pd.read_csv(path)

    print(f"    Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. Construcción del target
#    Reproduce las celdas 38-40 del notebook
# ─────────────────────────────────────────────────────────────────
def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Discretiza 'best' en tres clases de severidad usando los percentiles
    33 y 66 calculados sobre el dataset completo, igual que el notebook.
    0 = Bajo | 1 = Medio | 2 = Alto
    """
    q33 = df["best"].quantile(0.33)
    q66 = df["best"].quantile(0.66)
    print(f"    Umbrales severidad: q33={q33:.1f}  q66={q66:.1f}")

    def classify(v):
        if v <= q33:
            return 0   # Bajo
        elif v <= q66:
            return 1   # Medio
        else:
            return 2   # Alto

    return df["best"].map(classify).rename("severity")


# ─────────────────────────────────────────────────────────────────
# 3. Pipeline sklearn
#    Empaqueta los scalers para que sea serializable y reproducible
#    en la API de producción sin re-ejecutar celdas del notebook.
# ─────────────────────────────────────────────────────────────────
def build_sklearn_pipeline() -> Pipeline:
    """
    Construye el pipeline completo: features + escalado + clasificador.

    - UCDPFeatureBuilder: evento crudo -> 22 features (target encoding del
      país, frecuencia de díada, one-hot, temporales); aprende en fit.

    Reproduce las transformaciones de las celdas 43-56 del notebook:
    - SimpleImputer (mediana) + RobustScaler  para dyad_freq y country_te (sesgadas)
    - SimpleImputer (mediana) + StandardScaler para el resto de continuas
    - Passthrough para binarias / one-hot
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("robust",   Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  RobustScaler()),
            ]), SKEWED_FEATS),
            ("standard", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), STANDARD_FEATS),
            ("pass", "passthrough", PASSTHROUGH_FEATS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("features",     UCDPFeatureBuilder()),
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(**BEST_RF_PARAMS)),
    ])

    return pipeline


# ─────────────────────────────────────────────────────────────────
# 4. Entrenamiento
# ─────────────────────────────────────────────────────────────────
def train_and_log_model() -> str:
    # ── Configurar MLflow ────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[0/5] MLflow tracking URI: {MLFLOW_URI}")
    print(f"      Experimento        : {EXPERIMENT_NAME}")

    # ── Cargar y preparar datos ──────────────────────────────────
    df = load_data(DATA_PATH)

    print("[2/5] Construyendo target y split train/test ...")
    y = build_target(df)

    # Split estratificado idéntico al del notebook (celda 41)
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ── Entrenar y registrar en MLflow ───────────────────────────
    print("[4/4] Entrenando pipeline y guardando el artefacto en MLflow ...")
    with mlflow.start_run(run_name="rf_optimizado_notebook_params") as run:

        pipeline = build_sklearn_pipeline()
        pipeline.fit(X_raw_train[RAW_COLUMNS], y_train)

        # Métricas
        y_pred_train = pipeline.predict(X_raw_train[RAW_COLUMNS])
        y_pred_test  = pipeline.predict(X_raw_test[RAW_COLUMNS])

        f1_train  = f1_score(y_train, y_pred_train, average="macro")
        f1_test   = f1_score(y_test,  y_pred_test,  average="macro")
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test  = accuracy_score(y_test,  y_pred_test)

        # Log de parámetros
        mlflow.log_params(BEST_RF_PARAMS)
        mlflow.log_param("n_selected_features", len(SELECTED_FEATS))
        mlflow.log_param("selected_features",   ",".join(SELECTED_FEATS))

        # Log de métricas
        mlflow.log_metric("train_f1_macro", f1_train)
        mlflow.log_metric("test_f1_macro",  f1_test)
        mlflow.log_metric("train_accuracy", acc_train)
        mlflow.log_metric("test_accuracy",  acc_test)
        mlflow.log_metric("overfit_gap_f1", f1_train - f1_test)

        # Guardar reporte de clasificación como artefacto de texto
        report = classification_report(
            y_test, y_pred_test,
            target_names=["Bajo", "Medio", "Alto"]
        )
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt")

        # El DAG registra este artefacto después de que finaliza el entrenamiento.
        # skops sólo carga los tipos que considera seguros; el resto (el
        # transformer propio, el árbol de sklearn) se declara como confiable.
        trusted_types = skops.io.get_untrusted_types(data=skops.io.dumps(pipeline))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_raw_train[RAW_COLUMNS].head(3),
            signature=mlflow.models.infer_signature(
                X_raw_train[RAW_COLUMNS], y_pred_train
            ),
            # El módulo con el feature engineering viaja con el artefacto.
            code_paths=[os.path.join(os.path.dirname(__file__), "ucdp_features.py")],
            skops_trusted_types=trusted_types,
        )

        print(f"\n    run_id    : {run.info.run_id}")
        print(f"    f1_train  : {f1_train:.4f}")
        print(f"    f1_test   : {f1_test:.4f}")
        print(f"    acc_test  : {acc_test:.4f}")
        print(f"    gap F1    : {f1_train - f1_test:.4f}")
        print(f"\n{report}")

    return run.info.run_id
