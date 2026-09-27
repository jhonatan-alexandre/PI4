"""Build supervised sliding-window (lookback -> multi-horizon target) tf.data datasets."""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from . import config


def build_energy_targets(
    power_series: pd.Series | np.ndarray,
    lookback: int,
    horizons: dict[str, int] = config.HORIZONS,
) -> np.ndarray:
    """Sum future minute-average power into kWh, aligned after each input window."""
    values = np.asarray(power_series, dtype="float64")
    max_horizon = max(horizons.values())
    valid_windows = len(values) - lookback - max_horizon + 1
    if valid_windows <= 0:
        raise ValueError("The series must be longer than lookback plus the longest horizon.")

    cumulative_energy = np.concatenate(([0.0], np.cumsum(values))) / 60.0
    starts = np.arange(valid_windows) + lookback
    targets = np.column_stack([
        cumulative_energy[starts + horizon] - cumulative_energy[starts]
        for horizon in horizons.values()
    ])
    return targets.astype("float32")


def transform_energy_targets(targets: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(targets).astype("float32")


def make_dataset(
    df: pd.DataFrame,
    target_series: pd.Series,
    target_scaler: StandardScaler,
    lookback: int = config.LOOKBACK_MINUTES,
    horizons: dict[str, int] = config.HORIZONS,
    stride: int = config.WINDOW_STRIDE_MINUTES,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create (past feature window, future interval energy in kWh) batches."""
    features = df[config.FEATURE_COLUMNS].to_numpy(dtype="float32")
    targets = build_energy_targets(target_series, lookback, horizons)
    targets = transform_energy_targets(targets, target_scaler)

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets,
        sequence_length=lookback,
        sequence_stride=stride,
        batch_size=batch_size,
        shuffle=False,
    )
    if shuffle:
        dataset = dataset.shuffle(
            min(ceil(len(targets) / stride / batch_size), 128),
            seed=config.RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
    # sequence_stride keeps only every `stride`-th window, so the true sample
    # count (and therefore batch count) must divide the raw window count by it.
    n_strided_windows = ceil(len(targets) / stride)
    expected_batches = ceil(n_strided_windows / batch_size)
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(expected_batches))
    return dataset.prefetch(tf.data.AUTOTUNE)
