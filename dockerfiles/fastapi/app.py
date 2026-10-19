"""
app.py
======
API REST para servir el modelo de clasificación de severidad UCDP-GED.

Endpoints:
    GET  /                  → bienvenida
    GET  /health            → estado del servicio y del modelo
    GET  /model/info        → información del modelo cargado
    POST /model/reload      → recarga el modelo que esté en Production
    GET  /predict/example   → ejemplo de payload para /predict
    POST /predict           → predicción de severidad para un evento
    POST /predict/batch     → predicción para múltiples eventos
"""

import os
import threading
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
from typing import Literal, Optional
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME  = "ucdp-severity-rf"
MODEL_STAGE = "Production"

SEVERITY_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}

# Columnas crudas de UCDP-GED que recibe el modelo (el preprocesamiento
# vive dentro del Pipeline registrado en MLflow).
INPUT_COLUMNS = [
    "best", "deaths_civilians", "dyad_new_id", "latitude", "longitude",
    "date_start", "where_prec", "event_clarity", "date_prec", "country_id",
    "region", "type_of_violence",
]

# ─────────────────────────────────────────────────────────────────
# Carga del modelo al iniciar la API
# ─────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

model = None
model_loaded = False
model_version = None
model_run_id = None
_model_lock = threading.Lock()


def load_production_model():
    """
    Busca la versión en Production (la de número más alto si hubiera varias)
    y la carga por número de versión, para que el modelo servido y su
    metadata correspondan siempre a la misma versión.
    """
    candidates = [
        v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == MODEL_STAGE
    ]
    if not candidates:
        raise RuntimeError(f"No hay versiones de '{MODEL_NAME}' en {MODEL_STAGE}.")
    target = max(candidates, key=lambda v: int(v.version))
    loaded = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{target.version}")
    return loaded, str(target.version), target.run_id


def reload_model():
    """
    Recarga el modelo en Production. Si la carga falla, se mantiene el
    modelo anterior (si lo había) y se propaga la excepción.
    """
    global model, model_loaded, model_version, model_run_id
    with _model_lock:
        loaded, version, run_id = load_production_model()
        model, model_version, model_run_id = loaded, version, run_id
        model_loaded = True
    return version


try:
    reload_model()
    print(f"✅ Modelo '{MODEL_NAME}' v{model_version} ({MODEL_STAGE}) cargado correctamente.")
except Exception as e:
    print(f"⚠️  No se pudo cargar el modelo: {e}")

# ─────────────────────────────────────────────────────────────────
# Esquema de entrada
# ─────────────────────────────────────────────────────────────────
class EventFeatures(BaseModel):
    """Evento crudo, con los nombres de columna del UCDP-GED."""
    country_id:       int   = Field(...,              description="ID numérico del país (country_id de UCDP-GED)")
    dyad_new_id:      int   = Field(...,              description="ID de la díada (par de actores, dyad_new_id)")
    date_start:       date  = Field(...,              description="Fecha de inicio del evento (YYYY-MM-DD)")
    latitude:         float = Field(..., ge=-90, le=90,   description="Latitud del evento")
    longitude:        float = Field(..., ge=-180, le=180, description="Longitud del evento")
    region:           Literal["Africa", "Americas", "Asia", "Europe", "Middle East"] = Field(..., description="Región del evento")
    type_of_violence: Literal[1, 2, 3] = Field(...,   description="1=estatal, 2=no estatal, 3=unilateral")
    where_prec:       int   = Field(..., ge=1, le=7,  description="Precisión geográfica (1–7)")
    date_prec:        int   = Field(..., ge=1, le=5,  description="Precisión de la fecha (1–5)")
    event_clarity:    int   = Field(..., ge=1, le=2,  description="Claridad del evento (1=claro, 2=dudoso)")
    best:             Optional[float] = Field(None, ge=0, description="Estimación de muertes totales (opcional; con deaths_civilians arma civilian_ratio)")
    deaths_civilians: Optional[float] = Field(None, ge=0, description="Muertes civiles (opcional; si falta civilian_ratio = 0)")

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def features_to_df(event: EventFeatures) -> pd.DataFrame:
    data = event.model_dump()
    data["date_start"] = data["date_start"].isoformat()
    return pd.DataFrame([data])[INPUT_COLUMNS]

# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="UCDP-GED Severity Classifier",
    description=(
        "API REST para predecir la severidad (Bajo / Medio / Alto) de eventos "
        "de violencia organizada a partir del dataset UCDP-GED. "
        "Modelo: Random Forest optimizado con Macro F1 = 0.57 en test."
    ),
    version="1.0.0",
)

# ── GET / ─────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "UCDP-GED Severity Classifier API",
        "model":   f"{MODEL_NAME}/{MODEL_STAGE}",
        "docs":    "/docs",
    }

# ── GET /health ───────────────────────────────────────────────────
@app.get("/health", tags=["General"])
def health():
    """
    Verifica que la API está activa y que el modelo fue cargado correctamente.
    Siempre responde, independientemente del estado del modelo.
    """
    return {
        "status":        "ok" if model_loaded else "degraded",
        "model_loaded":  model_loaded,
        "model_name":    MODEL_NAME,
        "model_stage":   MODEL_STAGE,
        "model_version": model_version,
    }

# ── GET /model/info ───────────────────────────────────────────────
@app.get("/model/info", tags=["General"])
def model_info():
    """
    Devuelve información detallada del modelo en Production:
    versión, columnas de entrada y métricas registradas en MLflow.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verificá /health para más detalles."
        )
    try:
        # Metadata del modelo cargado en memoria, no de la última versión registrada.
        version, run_id = model_version, model_run_id
        run_data = client.get_run(run_id).data
        metrics = {k: round(v, 4) for k, v in run_data.metrics.items()}
        params  = run_data.params

        return {
            "model_name":    MODEL_NAME,
            "model_version": version,
            "model_stage":   MODEL_STAGE,
            "run_id":        run_id,
            "input_columns": INPUT_COLUMNS,
            "metrics":       metrics,
            "params":        params,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /model/reload ────────────────────────────────────────────
@app.post("/model/reload", tags=["General"])
def model_reload():
    """
    Vuelve a consultar MLflow y carga la versión que esté en Production.
    Lo invoca el DAG `ucdp_model_lifecycle` después de promover un modelo.
    Si la carga falla, la API sigue sirviendo el modelo anterior.
    """
    previous_version = model_version
    try:
        current_version = reload_model()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo recargar el modelo: {e}",
        )
    return {
        "previous_version": previous_version,
        "current_version":  current_version,
        "reloaded":         previous_version != current_version,
    }

# ── GET /predict/example ──────────────────────────────────────────
@app.get("/predict/example", tags=["Predicción"])
def predict_example():
    """
    Devuelve un ejemplo de payload válido para usar en POST /predict.
    Representa un evento de violencia estatal en Medio Oriente en 2015.
    """
    return {
        "country_id":       652,
        "dyad_new_id":      123,
        "date_start":       "2015-03-14",
        "latitude":         33.5,
        "longitude":        36.3,
        "region":           "Middle East",
        "type_of_violence": 1,
        "where_prec":       1,
        "date_prec":        1,
        "event_clarity":    1,
        "best":             10,
        "deaths_civilians": 5,
    }

# ── POST /predict ─────────────────────────────────────────────────
@app.post("/predict", tags=["Predicción"])
def predict(event: EventFeatures):
    """
    Predice la severidad de un único evento crudo de violencia organizada.

    Retorna la clase predicha (0=Bajo, 1=Medio, 2=Alto),
    su etiqueta en texto y las probabilidades por clase.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verificá /health para más detalles."
        )
    try:
        m     = model  # referencia fija por si hay un reload concurrente
        X     = features_to_df(event)
        pred  = int(m.predict(X)[0])
        proba = m.predict_proba(X)[0].tolist()

        return {
            "severity_code":  pred,
            "severity_label": SEVERITY_LABELS[pred],
            "probabilities":  {SEVERITY_LABELS[i]: round(p, 4) for i, p in enumerate(proba)},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /predict/batch ───────────────────────────────────────────
@app.post("/predict/batch", tags=["Predicción"])
def predict_batch(events: list[EventFeatures]):
    """
    Predice la severidad para una lista de eventos (máximo 100).

    Más eficiente que llamar a /predict en un loop cuando se tienen
    múltiples eventos para clasificar.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verificá /health para más detalles."
        )
    if len(events) > 100:
        raise HTTPException(
            status_code=422,
            detail="El batch no puede superar los 100 eventos por llamada."
        )
    try:
        m      = model  # referencia fija por si hay un reload concurrente
        X      = pd.concat([features_to_df(e) for e in events], ignore_index=True)
        preds  = m.predict(X).tolist()
        probas = m.predict_proba(X).tolist()

        return {
            "predictions": [
                {
                    "severity_code":  int(pred),
                    "severity_label": SEVERITY_LABELS[int(pred)],
                    "probabilities":  {SEVERITY_LABELS[i]: round(p, 4)
                                      for i, p in enumerate(proba)},
                }
                for pred, proba in zip(preds, probas)
            ],
            "count": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))