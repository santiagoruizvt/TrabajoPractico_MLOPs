"""
app.py
======
API REST para servir el modelo de clasificación de severidad UCDP-GED.

Endpoints:
    GET  /          → bienvenida
    GET  /health    → estado del servicio y del modelo
    POST /predict   → predicción de severidad para un evento
    POST /predict/batch → predicción para múltiples eventos
"""

import os
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME  = "ucdp-severity-rf"
MODEL_STAGE = "Production"

SEVERITY_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}

# ─────────────────────────────────────────────────────────────────
# Carga del modelo al iniciar la API
# ─────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    model_loaded = True
    print(f"✅ Modelo '{MODEL_NAME}/{MODEL_STAGE}' cargado correctamente.")
except Exception as e:
    model = None
    model_loaded = False
    print(f"⚠️  No se pudo cargar el modelo: {e}")

# ─────────────────────────────────────────────────────────────────
# Esquema de entrada (las 22 features seleccionadas)
# ─────────────────────────────────────────────────────────────────
class EventFeatures(BaseModel):
    civilian_ratio:     float = Field(..., ge=0, le=1,  description="Proporción de bajas civiles sobre el total (0–1)")
    country_te:         float = Field(...,               description="Target encoding del país")
    date_prec:          int   = Field(..., ge=1, le=5,  description="Precisión de la fecha del evento (1–5)")
    dyad_freq:          float = Field(..., ge=0,         description="Frecuencia histórica del par de actores")
    dyad_id:            int   = Field(...,               description="ID numérico de la díada")
    era_1989_99:        int   = Field(..., ge=0, le=1,  alias="era_1989-99",  description="1 si el evento ocurrió entre 1989 y 1999")
    era_2000_09:        int   = Field(..., ge=0, le=1,  alias="era_2000-09",  description="1 si el evento ocurrió entre 2000 y 2009")
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
# Helpers
# ─────────────────────────────────────────────────────────────────
FEATURE_ORDER = sorted([
    "civilian_ratio", "country_te",   "date_prec",    "dyad_freq",
    "dyad_id",        "era_1989-99",  "era_2000-09",  "event_clarity",
    "latitude",       "longitude",    "month_cos",    "month_sin",
    "region_Africa",  "region_Americas", "region_Asia", "region_Europe",
    "region_Middle East", "tov_1",    "tov_2",        "tov_3",
    "where_prec",     "year",
])

def features_to_df(event: EventFeatures) -> pd.DataFrame:
    """Convierte el objeto Pydantic a un DataFrame con el orden canónico de features."""
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
    """
    return {
        "status":       "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name":   MODEL_NAME,
        "model_stage":  MODEL_STAGE,
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
        X = features_to_df(event)
        pred        = int(model.predict(X)[0])
        proba       = model.predict_proba(X)[0].tolist()
        proba_dict  = {SEVERITY_LABELS[i]: round(p, 4) for i, p in enumerate(proba)}

        return {
            "severity_code":  pred,
            "severity_label": SEVERITY_LABELS[pred],
            "probabilities":  proba_dict,
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
        X = pd.concat([features_to_df(e) for e in events], ignore_index=True)
        preds  = model.predict(X).tolist()
        probas = model.predict_proba(X).tolist()

        results = []
        for pred, proba in zip(preds, probas):
            results.append({
                "severity_code":  int(pred),
                "severity_label": SEVERITY_LABELS[int(pred)],
                "probabilities":  {SEVERITY_LABELS[i]: round(p, 4)
                                   for i, p in enumerate(proba)},
            })

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))