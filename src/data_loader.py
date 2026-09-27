"""Fetch and cache the UCI Individual Household Electric Power Consumption dataset (id=235)."""
from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

from . import config


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """Return the raw dataset as a single DataFrame, downloading and caching it on first use."""
    if config.RAW_CSV_PATH.exists() and not force_download:
        return pd.read_csv(config.RAW_CSV_PATH, low_memory=False)

    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets

    if targets is not None and not targets.empty:
        df = pd.concat([features, targets], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        df = features

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV_PATH, index=False)
    return df
