"""LSTM architecture for multi-horizon energy consumption forecasting."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config


def build_lstm_model(
    n_features: int,
    n_outputs: int,
    lookback: int = config.LOOKBACK_MINUTES,
    lstm_units: tuple[int, ...] = config.LSTM_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Stacked-LSTM regressor with one output per forecast horizon."""
    inputs = keras.Input(shape=(lookback, n_features), name="lookback_window")
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = layers.LSTM(units, return_sequences=return_sequences)(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_outputs, name="horizon_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_energy_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
