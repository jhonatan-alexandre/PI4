"""Causal features and observed-only energy labels on an intact minute grid.

An origin is the first minute to forecast. Every input ends at origin - 1.
Gaps remain NaN: never interpolate a label or compress time by dropping rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def partition_origins(start: int, stop: int, lookback: int = 120,
                      stride: int = 5) -> np.ndarray:
    """Keep the entire input and every target inside this partition."""
    if stride <= 0 or lookback <= 0 or stop <= start:
        raise ValueError("Invalid partition, lookback or stride.")
    return np.arange(start + lookback, stop - max(config.HORIZONS.values()) + 1, stride)


def make_samples(df: pd.DataFrame, origins: np.ndarray, lookback: int = 120,
                 bin_minutes: int = 5) -> dict:
    """Build five-minute mean sequences plus calendar/rolling/seasonal context.

    Optional daily/weekly lags can be absent; a training-only imputer handles
    those. A missing measurement in the input window or target rejects the
    sample, including partially observed aggregation bins.
    """
    if lookback <= 0 or bin_minutes <= 0 or lookback % bin_minutes:
        raise ValueError("lookback must be a positive multiple of bin_minutes.")
    if (not isinstance(df.index, pd.DatetimeIndex) or
            not df.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all()):
        raise ValueError("Measurements must have a continuous, unique minute index.")
    origins = np.asarray(origins, dtype=np.int64)
    horizon = max(config.HORIZONS.values())
    if len(origins) == 0 or origins.min() < lookback or origins.max() + horizon > len(df):
        raise ValueError("No usable origins, or input/target extends outside the series.")
    values = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    power = df[config.TARGET_COLUMN]
    bad_inputs = np.r_[0, (~np.isfinite(values).all(axis=1)).cumsum()]
    bad_targets = np.r_[0, (~np.isfinite(values[:, 0])).cumsum()]
    valid = ((bad_inputs[origins] == bad_inputs[origins - lookback]) &
             (bad_targets[origins + horizon] == bad_targets[origins]))
    origins = origins[valid]
    if not len(origins):
        raise ValueError("All windows intersect missing measurements.")
    ends = origins - 1
    energy = np.r_[0., np.nan_to_num(values[:, 0], nan=0).cumsum()] / 60.
    y = np.column_stack([energy[origins + h] - energy[origins]
                         for h in config.HORIZONS.values()]).astype("float32")
    averaged = df[config.FEATURE_COLUMNS].rolling(bin_minutes).mean().to_numpy("float32")
    offsets = np.arange(-lookback + bin_minutes - 1, 0, bin_minutes)
    sequence = averaged[origins[:, None] + offsets]

    context = {c: values[ends, j] for j, c in enumerate(config.FEATURE_COLUMNS)}
    for window in (5, 15, 30, 60, 120):
        context[f"power_mean_{window}m"] = power.rolling(window).mean().iloc[ends].to_numpy()
    for window in (15, 60):
        context[f"power_std_{window}m"] = power.rolling(window).std().iloc[ends].to_numpy()
    context["power_min_60m"] = power.rolling(60).min().iloc[ends].to_numpy()
    context["power_max_60m"] = power.rolling(60).max().iloc[ends].to_numpy()
    for lag in (5, 15, 1440, 10080):
        context[f"power_lag_{lag}m"] = power.shift(lag).iloc[ends].to_numpy()
    for lag, label in ((1440, "day"), (10080, "week")):
        for name, h in config.HORIZONS.items():
            # Energy at the SAME future clock interval on the previous day/week.
            context[f"seasonal_{label}_{name}"] = (
                power.rolling(h).sum().shift(lag - h).iloc[ends].to_numpy() / 60.)
    times = df.index[origins]
    for name, phase in (
        ("hour", (times.hour * 60 + times.minute) / 1440),
        ("weekday", times.dayofweek / 7),
        ("year", (times.dayofyear - 1) / 365.25),
    ):
        context[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        context[f"{name}_cos"] = np.cos(2 * np.pi * phase)
    context["weekend"] = (times.dayofweek >= 5).astype(float)
    context["unmetered_power_kw"] = (
        values[ends, 0] - values[ends, 4:7].sum(axis=1) * 60 / 1000)
    table = pd.DataFrame(context, index=times).astype("float32")
    return {"origins": origins, "times": times, "sequence": sequence,
            "context": table.to_numpy(), "context_columns": list(table.columns), "y": y}


def subset(samples: dict, mask: np.ndarray) -> dict:
    """Select aligned sample arrays without altering column metadata."""
    return {key: (value if key == "context_columns" else value[mask])
            for key, value in samples.items()}
