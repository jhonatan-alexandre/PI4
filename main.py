"""End-to-end pipeline: fetch -> clean -> window -> scale -> train -> evaluate.

Usage:
    python main.py [--epochs 30] [--lookback 120] [--max-rows 200000] [--force-download]
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from src import config, data_loader, evaluate, model as model_module, preprocessing, scaling, train, windowing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lookback", type=int, default=config.LOOKBACK_MINUTES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=config.MAX_ROWS)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.MAX_ROWS = args.max_rows

    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("1/5 - Carregando dados brutos (cache local se já baixado)...")
    raw_df = data_loader.load_raw_data(force_download=args.force_download)

    print("2/5 - Limpando e preparando a série temporal...")
    clean_df = preprocessing.build_clean_dataset(raw_df)
    train_df, val_df, test_df = preprocessing.chronological_split(clean_df)
    print(f"   treino={len(train_df)}  val={len(val_df)}  teste={len(test_df)} amostras (1 min/linha)")

    print("3/5 - Normalizando features e alvos kWh (scalers ajustados no treino)...")
    scaler = scaling.fit_scaler(train_df)
    energy_scaler = scaling.fit_energy_target_scaler(
        train_df, lookback=args.lookback, stride=config.WINDOW_STRIDE_MINUTES
    )
    train_scaled = scaling.apply_scaler(train_df, scaler)
    val_scaled = scaling.apply_scaler(val_df, scaler)
    test_scaled = scaling.apply_scaler(test_df, scaler)

    print("4/5 - Construindo janelas deslizantes e treinando a LSTM...")
    train_ds = windowing.make_dataset(
        train_scaled,
        target_series=train_df[config.TARGET_COLUMN],
        target_scaler=energy_scaler,
        lookback=args.lookback,
        stride=config.WINDOW_STRIDE_MINUTES,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_ds = windowing.make_dataset(
        val_scaled,
        target_series=val_df[config.TARGET_COLUMN],
        target_scaler=energy_scaler,
        lookback=args.lookback,
        stride=config.WINDOW_STRIDE_MINUTES,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = windowing.make_dataset(
        test_scaled,
        target_series=test_df[config.TARGET_COLUMN],
        target_scaler=energy_scaler,
        lookback=args.lookback,
        stride=config.WINDOW_STRIDE_MINUTES,
        batch_size=args.batch_size,
        shuffle=False,
    )

    lstm_model = model_module.build_lstm_model(
        n_features=len(config.FEATURE_COLUMNS),
        n_outputs=len(config.HORIZONS),
        lookback=args.lookback,
    )
    lstm_model.summary()
    train.train_model(lstm_model, train_ds, val_ds, epochs=args.epochs)

    print("5/5 - Avaliando no conjunto de teste e gerando gráficos/métricas...")
    metrics_df = evaluate.evaluate_model(
        lstm_model,
        test_ds,
        target_center=energy_scaler.mean_,
        target_scale=energy_scaler.scale_,
    )
    print(metrics_df.to_string(index=False))
    print(f"\nMétricas salvas em: {config.METRICS_DIR}")
    print(f"Gráficos salvos em: {config.FIGURES_DIR}")
    print(f"Modelo salvo em: {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
