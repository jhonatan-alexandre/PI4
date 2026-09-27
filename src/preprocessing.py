"""Cleaning, resampling and chronological splitting of the household power dataset."""
from __future__ import annotations

import pandas as pd

from . import config


def parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DatetimeIndex from the raw Date/Time columns and sort chronologically."""
    df = df.copy()
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"])
    df.index = dt
    df = df.sort_index()
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert measurement columns to float, turning the '?' missing markers into NaN."""
    df = df.copy()
    for col in config.FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill short gaps with time-based interpolation; drop any still-missing rows at the edges."""
    df = df.copy()
    df[config.FEATURE_COLUMNS] = (
        df[config.FEATURE_COLUMNS].interpolate(method="time", limit_direction="both")
    )
    df = df.dropna(subset=config.FEATURE_COLUMNS)
    return df


def build_clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: parse dates, coerce numerics, interpolate, cache as parquet."""
    if config.MAX_ROWS is not None and config.MAX_ROWS <= 0:
        raise ValueError("MAX_ROWS must be positive or None.")
    if config.CLEAN_PARQUET_PATH.exists():
        cached = pd.read_parquet(config.CLEAN_PARQUET_PATH)
        # Older versions cached a truncated tail and silently ignored later limits.
        if len(cached) == len(raw_df):
            return cached if config.MAX_ROWS is None else cached.tail(config.MAX_ROWS)

    df = parse_datetime_index(raw_df)
    df = clean_numeric_columns(df)
    df = fill_missing(df)
    df = df[config.FEATURE_COLUMNS]

    # Cache the complete series; apply the experiment's limit only on return.
    df.to_parquet(config.CLEAN_PARQUET_PATH)
    return df if config.MAX_ROWS is None else df.tail(config.MAX_ROWS)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test without shuffling."""
    n = len(df)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = train_end + int(n * config.VAL_FRAC)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df
