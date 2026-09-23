"""
init_mlflow_experiment.py
=========================
Script de inicialización único (one-shot) para el proyecto UCDP-GED.

  1. Lee el dataset UCDP-GED desde MinIO (s3://data/ucdp_ged.csv)
  2. Reproduce exactamente el preprocesamiento del notebook (feature engineering,
     encoders, scalers, selección de features) pero empaquetado en un Pipeline
     sklearn serializable.
  3. Crea el experimento en MLflow y registra una primera run con el
     Random Forest optimizado (mejores hiperparámetros del notebook).
  4. Promueve el modelo al stage "Production" en el Model Registry.

Ejecutar UNA sola vez, desde el host o desde un contenedor con acceso a MLflow y MinIO:
    python init_mlflow_experiment.py

"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import category_encoders as ce

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0. Configuración
# ─────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI",    "http://localhost:5000")
DATA_PATH       = os.getenv("DATA_BUCKET",             "s3://data/ucdp_ged.csv")
MODEL_NAME      = "ucdp-severity-rf"
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

# Features seleccionadas por ANOVA + varianza (22 de 23, la descartada es
# 'era_2010-23' por p-value > 0.05, reproduciendo la celda 65 del notebook)
SELECTED_FEATS = sorted([
    "civilian_ratio", "country_te",   "date_prec",    "dyad_freq",
    "dyad_id",        "era_1989-99",  "era_2000-09",  "event_clarity",
    "latitude",       "longitude",    "month_cos",    "month_sin",
    "region_Africa",  "region_Americas", "region_Asia", "region_Europe",
    "region_Middle East", "tov_1",    "tov_2",        "tov_3",
    "where_prec",     "year",
])

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
# 3. Feature Engineering
#    Reproduce build_features() de la celda 52 del notebook.
#    Devuelve un DataFrame con exactamente SELECTED_FEATS.
# ─────────────────────────────────────────────────────────────────
def build_features(df_src: pd.DataFrame,
                   country_encoder: ce.TargetEncoder = None,
                   fit_encoder: bool = False,
                   y: pd.Series = None) -> pd.DataFrame:
    """
    Construye la matriz de features replicando la lógica del notebook.

    Parámetros
    ----------
    df_src        : DataFrame original (con todas las columnas UCDP)
    country_encoder: TargetEncoder ya ajustado (para transform-only en test)
    fit_encoder   : True sólo en train — ajusta el TargetEncoder
    y             : target necesario para ajustar el TargetEncoder

    Retorna
    -------
    fe            : DataFrame con SELECTED_FEATS en orden canónico
    country_encoder: encoder ajustado (para reusar en test)
    """
    fe = pd.DataFrame(index=df_src.index)

    # Ratio de civiles (anti-leakage: no usa 'best' directamente)
    best_safe         = df_src["best"].replace(0, np.nan)
    fe["civilian_ratio"] = (
        df_src["deaths_civilians"] / best_safe
    ).fillna(0).clip(0, 1)

    # Díada (par de actores del conflicto)
    fe["dyad_id"] = df_src["dyad_new_id"]

    # Geo
    fe["latitude"]  = df_src["latitude"]
    fe["longitude"] = df_src["longitude"]

    # Temporal: mes cíclico + año
    month           = pd.to_datetime(df_src["date_start"]).dt.month
    fe["month_sin"] = np.sin(2 * np.pi * month / 12)
    fe["month_cos"] = np.cos(2 * np.pi * month / 12)
    fe["year"]      = pd.to_datetime(df_src["date_start"]).dt.year

    # Precisión geográfica y temporal
    fe["where_prec"]    = df_src["where_prec"].map(
        {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
    ).fillna(df_src["where_prec"])
    fe["event_clarity"] = df_src["event_clarity"]
    fe["date_prec"]     = df_src["date_prec"]

    # Frecuencia de la díada (frequency encoding sobre train)
    dyad_freq           = df_src["dyad_new_id"].map(
        df_src["dyad_new_id"].value_counts()
    )
    fe["dyad_freq"] = dyad_freq

    # Target encoding del país (ajustar SOLO en train)
    if fit_encoder:
        country_encoder = ce.TargetEncoder(cols=["country_id"])
        te_input        = df_src[["country_id"]].copy()
        fe["country_te"] = country_encoder.fit_transform(
            te_input, y
        )["country_id"]
    else:
        te_input         = df_src[["country_id"]].copy()
        fe["country_te"] = country_encoder.transform(te_input)["country_id"]

    # One-hot región
    region_dummies = pd.get_dummies(df_src["region"], prefix="region")
    for col in ["region_Africa", "region_Americas", "region_Asia",
                "region_Europe", "region_Middle East"]:
        fe[col] = region_dummies.get(col, 0).astype(int)

    # Type of violence (one-hot)
    for v in [1, 2, 3]:
        fe[f"tov_{v}"] = (df_src["type_of_violence"] == v).astype(int)

    # Eras históricas
    fe["era_1989-99"] = (fe["year"] < 2000).astype(int)
    fe["era_2000-09"] = ((fe["year"] >= 2000) & (fe["year"] < 2010)).astype(int)
    # era_2010-23 se descarta (p-value ANOVA > 0.05, celda 65 del notebook)

    # Orden canónico de SELECTED_FEATS
    fe = fe[SELECTED_FEATS]

    return fe, country_encoder


# ─────────────────────────────────────────────────────────────────
# 4. Pipeline sklearn
#    Empaqueta los scalers para que sea serializable y reproducible
#    en la API de producción sin re-ejecutar celdas del notebook.
# ─────────────────────────────────────────────────────────────────
def build_sklearn_pipeline() -> Pipeline:
    """
    Construye el pipeline de preprocesamiento + clasificador.

    Reproduce las transformaciones de las celdas 43-56 del notebook:
    - SimpleImputer (mediana) + RobustScaler  para dyad_freq y country_te (sesgadas)
    - SimpleImputer (mediana) + StandardScaler para el resto de continuas
    - Passthrough para binarias / one-hot

    El TargetEncoder y el frequency encoding están fuera del pipeline
    porque dependen del target y del conteo de filas de train; se
    aplican en build_features() antes de entrar al pipeline.
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
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(**BEST_RF_PARAMS)),
    ])

    return pipeline


# ─────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Configurar MLflow ────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
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

    print("[3/5] Feature engineering ...")
    X_train_fe, country_encoder = build_features(
        X_raw_train, fit_encoder=True, y=y_train
    )
    X_test_fe, _ = build_features(
        X_raw_test, country_encoder=country_encoder, fit_encoder=False
    )

    # ── Entrenar y registrar en MLflow ───────────────────────────
    print("[4/5] Entrenando pipeline y registrando en MLflow ...")
    with mlflow.start_run(run_name="rf_optimizado_notebook_params") as run:

        pipeline = build_sklearn_pipeline()
        pipeline.fit(X_train_fe, y_train)

        # Métricas
        y_pred_train = pipeline.predict(X_train_fe)
        y_pred_test  = pipeline.predict(X_test_fe)

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

        # Registrar el pipeline completo en el Model Registry
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train_fe.head(3),
            signature=mlflow.models.infer_signature(
                X_train_fe, y_pred_train
            ),
        )

        print(f"\n    run_id    : {run.info.run_id}")
        print(f"    f1_train  : {f1_train:.4f}")
        print(f"    f1_test   : {f1_test:.4f}")
        print(f"    acc_test  : {acc_test:.4f}")
        print(f"    gap F1    : {f1_train - f1_test:.4f}")
        print(f"\n{report}")

    # ── Promover a Production en el Model Registry ───────────────
    print("[5/5] Promoviendo modelo a stage 'Production' ...")
    # La versión recién registrada es la última
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])
    version = latest[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,   # archiva versiones anteriores
    )
    print(f"    Modelo '{MODEL_NAME}' v{version} → Production ✓")
    print("\n✅ Inicialización completa.")
    print(f"   Cargá el modelo en la API con:")
    print(f"   mlflow.sklearn.load_model('models:/{MODEL_NAME}/Production')")


if __name__ == "__main__":
    main()