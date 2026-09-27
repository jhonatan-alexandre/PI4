"""Feature scaling utilities (fit on train split only, reused for val/test)."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config
from .windowing import build_energy_targets

SCALER_PATH = config.MODELS_DIR / "feature_scaler.joblib"
ENERGY_SCALER_PATH = config.MODELS_DIR / "energy_target_scaler.joblib"


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURE_COLUMNS])
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def fit_energy_target_scaler(
    train_df: pd.DataFrame,
    lookback: int = config.LOOKBACK_MINUTES,
    stride: int = config.WINDOW_STRIDE_MINUTES,
) -> StandardScaler:
    targets_kwh = build_energy_targets(
        train_df[config.TARGET_COLUMN],
        lookback=lookback,
        horizons=config.HORIZONS,
    )
    scaler = StandardScaler().fit(targets_kwh[::stride])
    joblib.dump(scaler, ENERGY_SCALER_PATH)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[config.FEATURE_COLUMNS] = scaler.transform(df[config.FEATURE_COLUMNS])
    return df
