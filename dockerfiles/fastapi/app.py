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
from typing import Optional
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME  = "ucdp-severity-rf"
MODEL_STAGE = "Production"

SEVERITY_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}

FEATURE_ORDER = sorted([
    "civilian_ratio", "country_te",   "date_prec",    "dyad_freq",
    "dyad_id",        "era_1989-99",  "era_2000-09",  "event_clarity",
    "latitude",       "longitude",    "month_cos",    "month_sin",
    "region_Africa",  "region_Americas", "region_Asia", "region_Europe",
    "region_Middle East", "tov_1",    "tov_2",        "tov_3",
    "where_prec",     "year",
])

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
    civilian_ratio:     float = Field(..., ge=0, le=1,  description="Proporción de bajas civiles sobre el total (0–1)")
    country_te:         float = Field(...,               description="Target encoding del país")
    date_prec:          int   = Field(..., ge=1, le=5,  description="Precisión de la fecha del evento (1–5)")
    dyad_freq:          float = Field(..., ge=0,         description="Frecuencia histórica del par de actores")
    dyad_id:            int   = Field(...,               description="ID numérico de la díada")
    era_1989_99:        int   = Field(..., ge=0, le=1,  alias="era_1989-99",       description="1 si el evento ocurrió entre 1989 y 1999")
    era_2000_09:        int   = Field(..., ge=0, le=1,  alias="era_2000-09",       description="1 si el evento ocurrió entre 2000 y 2009")
    event_clarity:      int   = Field(..., ge=1, le=2,  description="Claridad del evento (1=claro, 2=dudoso)")
    latitude:           float = Field(...,               description="Latitud geográfica del evento")
    longitude:          float = Field(...,               description="Longitud geográfica del evento")
    month_cos:          float = Field(..., ge=-1, le=1, description="Componente coseno del mes (encoding cíclico)")
    month_sin:          float = Field(..., ge=-1, le=1, description="Componente seno del mes (encoding cíclico)")
    region_Africa:      int   = Field(..., ge=0, le=1,  description="1 si el evento ocurrió en África")
    region_Americas:    int   = Field(..., ge=0, le=1,  description="1 si el evento ocurrió en América")
    region_Asia:        int   = Field(..., ge=0, le=1,  description="1 si el evento ocurrió en Asia")
    region_Europe:      int   = Field(..., ge=0, le=1,  description="1 si el evento ocurrió en Europa")
    region_Middle_East: int   = Field(..., ge=0, le=1,  alias="region_Middle East", description="1 si el evento ocurrió en Medio Oriente")
    tov_1:              int   = Field(..., ge=0, le=1,  description="1 si es violencia estatal")
    tov_2:              int   = Field(..., ge=0, le=1,  description="1 si es violencia no estatal")
    tov_3:              int   = Field(..., ge=0, le=1,  description="1 si es violencia unilateral")
    where_prec:         int   = Field(..., ge=1, le=7,  description="Precisión geográfica del evento (1–7)")
    year:               int   = Field(..., ge=1989,      description="Año del evento")

    class Config:
        populate_by_name = True

# ─────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────
def features_to_df(event: EventFeatures) -> pd.DataFrame:
    data = event.model_dump(by_alias=True)
    return pd.DataFrame([data])[FEATURE_ORDER]

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
    versión, features esperadas y métricas registradas en MLflow.
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
            "features":      FEATURE_ORDER,
            "n_features":    len(FEATURE_ORDER),
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
        "civilian_ratio":    0.5,
        "country_te":        1.2,
        "date_prec":         1,
        "dyad_freq":         100.0,
        "dyad_id":           123,
        "era_1989-99":       0,
        "era_2000-09":       0,
        "event_clarity":     1,
        "latitude":          33.5,
        "longitude":         36.3,
        "month_cos":         0.5,
        "month_sin":         0.8,
        "region_Africa":     0,
        "region_Americas":   0,
        "region_Asia":       0,
        "region_Europe":     0,
        "region_Middle East": 1,
        "tov_1":             1,
        "tov_2":             0,
        "tov_3":             0,
        "where_prec":        1,
        "year":              2015,
    }

# ── POST /predict ─────────────────────────────────────────────────
@app.post("/predict", tags=["Predicción"])
def predict(event: EventFeatures):
    """
    Predice la severidad de un único evento de violencia organizada.

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