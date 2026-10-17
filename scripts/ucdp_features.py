"""
ucdp_features.py
================
Feature engineering del modelo de severidad UCDP-GED como transformer sklearn.

Vive dentro del Pipeline para que el modelo registrado en MLflow reciba el
evento CRUDO (columnas de UCDP-GED) y aplique solo todo el preprocesamiento:
target encoding del país, frecuencia de díada, one-hot y features temporales.
Los estadísticos aprendidos (conteos de díada, encoder del país) se ajustan
en `fit` con el set de entrenamiento y se serializan con el modelo.

El entrenamiento lo registra con `code_paths` para que este módulo viaje con
el artefacto y la API pueda deserializarlo sin copiarlo en su imagen.
"""

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin

# Columnas crudas que el modelo espera recibir.
RAW_COLUMNS = [
    "best", "deaths_civilians", "dyad_new_id", "latitude", "longitude",
    "date_start", "where_prec", "event_clarity", "date_prec", "country_id",
    "region", "type_of_violence",
]

REGIONS = ["Africa", "Americas", "Asia", "Europe", "Middle East"]

# Features finales (22 de 23; 'era_2010-23' se descarta por ANOVA p > 0.05).
SELECTED_FEATS = sorted([
    "civilian_ratio", "country_te",   "date_prec",    "dyad_freq",
    "dyad_id",        "era_1989-99",  "era_2000-09",  "event_clarity",
    "latitude",       "longitude",    "month_cos",    "month_sin",
    "region_Africa",  "region_Americas", "region_Asia", "region_Europe",
    "region_Middle East", "tov_1",    "tov_2",        "tov_3",
    "where_prec",     "year",
])


class UCDPFeatureBuilder(BaseEstimator, TransformerMixin):
    """Evento crudo UCDP-GED -> DataFrame con SELECTED_FEATS (sin escalar)."""

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("UCDPFeatureBuilder necesita y para el target encoding.")
        X = self._as_frame(X)
        self.dyad_counts_ = X["dyad_new_id"].value_counts()
        self.country_encoder_ = ce.TargetEncoder(cols=["country_id"])
        self.country_encoder_.fit(X[["country_id"]], y)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        fe = pd.DataFrame(index=X.index)

        # Ratio de civiles; sin 'best' o con best == 0 queda en 0.
        best = pd.to_numeric(X["best"], errors="coerce").replace(0, np.nan)
        civ = pd.to_numeric(X["deaths_civilians"], errors="coerce")
        fe["civilian_ratio"] = (civ / best).fillna(0).clip(0, 1)

        fe["dyad_id"] = X["dyad_new_id"]
        fe["latitude"] = X["latitude"]
        fe["longitude"] = X["longitude"]

        date = pd.to_datetime(X["date_start"])
        month = date.dt.month
        fe["month_sin"] = np.sin(2 * np.pi * month / 12)
        fe["month_cos"] = np.cos(2 * np.pi * month / 12)
        fe["year"] = date.dt.year

        fe["where_prec"] = pd.to_numeric(X["where_prec"], errors="coerce")
        fe["event_clarity"] = X["event_clarity"]
        fe["date_prec"] = X["date_prec"]

        # Díada no vista en train -> NaN (lo imputa la mediana del ColumnTransformer).
        fe["dyad_freq"] = X["dyad_new_id"].map(self.dyad_counts_)
        fe["country_te"] = self.country_encoder_.transform(
            X[["country_id"]]
        )["country_id"]

        for region in REGIONS:
            fe[f"region_{region}"] = (X["region"] == region).astype(int)
        for v in (1, 2, 3):
            fe[f"tov_{v}"] = (X["type_of_violence"] == v).astype(int)

        fe["era_1989-99"] = (fe["year"] < 2000).astype(int)
        fe["era_2000-09"] = ((fe["year"] >= 2000) & (fe["year"] < 2010)).astype(int)

        return fe[SELECTED_FEATS]

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in RAW_COLUMNS:
            if col not in X.columns:
                X[col] = np.nan
        return X
