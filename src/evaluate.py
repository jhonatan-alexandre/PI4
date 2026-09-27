"""Metrics computation and report-ready plots for the trained forecaster."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow import keras

from . import config

INK = "#18332F"
MUTED = "#64736F"
PAPER = "#F7F6F1"
GRID = "#DDE2DB"
TEAL = "#087F78"
CORAL = "#D86B50"
GOLD = "#C19335"
HORIZON_COLORS = [TEAL, CORAL, GOLD]


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _style_figure(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.patch.set_facecolor(PAPER)
    fig.suptitle(title, x=0.075, y=0.99, ha="left", color=INK,
                 fontsize=19, fontweight="bold", fontfamily="Georgia")
    fig.text(0.075, 0.94, subtitle, ha="left", color=MUTED, fontsize=10)


def collect_predictions(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a full dataset and return (y_true, y_pred), shape (n_samples, n_horizons)."""
    y_true_batches, y_pred_batches = [], []
    for x_batch, y_batch in dataset:
        y_true_batches.append(y_batch.numpy())
        y_pred_batches.append(model(x_batch, training=False).numpy())
    return np.concatenate(y_true_batches), np.concatenate(y_pred_batches)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: dict[str, int] = config.HORIZONS
) -> pd.DataFrame:
    """Per-horizon MAE, RMSE, MAPE and R2, returned as a tidy DataFrame."""
    rows = []
    for i, name in enumerate(horizons):
        true_i, pred_i = y_true[:, i], y_pred[:, i]
        mae = mean_absolute_error(true_i, pred_i)
        rmse = float(np.sqrt(np.mean((true_i - pred_i) ** 2)))
        mape = mean_absolute_percentage_error(true_i, pred_i) * 100
        r2 = r2_score(true_i, pred_i)
        rows.append({"horizonte": name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2})
    return pd.DataFrame(rows)


def save_metrics(metrics_df: pd.DataFrame, filename: str = "test_metrics") -> None:
    metrics_df.to_csv(config.METRICS_DIR / f"{filename}.csv", index=False)
    with open(config.METRICS_DIR / f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)


def plot_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Create a styled, report-ready table of evaluation metrics by horizon."""
    columns = ["horizonte", "MAE", "RMSE", "MAPE_%", "R2"]
    headings = ["INTERVALO", "MAE · kWh", "RMSE · kWh", "MAPE", "R²"]
    values = []
    for row in metrics_df[columns].itertuples(index=False, name=None):
        values.append([
            row[0].upper(),
            f"{row[1]:.3f}",
            f"{row[2]:.3f}",
            f"{row[3]:.2f}%",
            f"{row[4]:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10.5, 2.6 + 0.65 * len(values)))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.axis("off")
    fig.suptitle("Desempenho por horizonte", x=0.075, y=0.96, ha="left", color=INK,
                 fontsize=19, fontweight="bold", fontfamily="Georgia")
    fig.text(0.075, 0.88, "Resultados no conjunto de teste · energia acumulada no intervalo futuro",
             ha="left", color=MUTED, fontsize=10)

    table = ax.table(
        cellText=values,
        colLabels=headings,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.24, 0.18, 0.18, 0.18, 0.18],
        bbox=[0.035, 0.24, 0.93, 0.53],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row_index == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="#FFFFFF", weight="bold", fontsize=9)
            cell.set_height(0.15)
        else:
            cell.set_facecolor("#E9EFEB" if row_index % 2 else "#FFFFFF")
            cell.set_text_props(color=INK, weight="bold" if col_index == 0 else "normal")
            cell.set_height(0.16)
            if col_index == 0:
                cell.get_text().set_color(HORIZON_COLORS[(row_index - 1) % len(HORIZON_COLORS)])

    fig.text(0.075, 0.12, "MAE/RMSE: erro em kWh  ·  MAPE: erro percentual  ·  R²: ajuste (mais próximo de 1 é melhor)",
             ha="left", color=MUTED, fontsize=8.5)
    fig.text(0.075, 0.075, "Avaliação realizada sobre dados não usados no treinamento",
             ha="left", color=MUTED, fontsize=8.5, style="italic")
    fig.savefig(config.FIGURES_DIR / "metrics_summary.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_experiment_summary(df: pd.DataFrame) -> None:
    """Create a technical-summary figure describing the data and trained network setup."""
    sample_count = len(df)
    train_end = int(sample_count * config.TRAIN_FRAC)
    val_end = train_end + int(sample_count * config.VAL_FRAC)
    split_rows = [
        ("Treino", df.index[0], df.index[train_end - 1], train_end),
        ("Validação", df.index[train_end], df.index[val_end - 1], val_end - train_end),
        ("Teste", df.index[val_end], df.index[-1], sample_count - val_end),
    ]
    date_start = df.index.min().strftime("%d/%m/%Y")
    date_end = df.index.max().strftime("%d/%m/%Y")
    total_share = sample_count / config.UCI_TOTAL_ROWS * 100

    fig = plt.figure(figsize=(13.5, 9.4))
    fig.patch.set_facecolor(PAPER)
    fig.suptitle("Ficha técnica do experimento", x=0.065, y=0.965, ha="left",
                 color=INK, fontsize=22, fontweight="bold", fontfamily="Georgia")
    fig.text(0.066, 0.922, "Previsão de consumo residencial de energia · rede LSTM",
             ha="left", color=MUTED, fontsize=11)

    from matplotlib.patches import Rectangle

    fig.add_artist(Rectangle((0.055, 0.765), 0.89, 0.12, transform=fig.transFigure,
                             facecolor=INK, edgecolor="none", zorder=0))
    top_items = [
        ("AMOSTRAS ANALISADAS", f"{sample_count:,}".replace(",", ".")),
        ("PERÍODO DA SÉRIE", f"{date_start}  —  {date_end}"),
        ("FREQUÊNCIA", "1 registro / minuto"),
        ("DIVISÃO TEMPORAL", "70%  ·  15%  ·  15%"),
    ]
    for i, (label, value) in enumerate(top_items):
        x = 0.078 + i * 0.218
        fig.text(x, 0.842, label, color="#B9D1C7", fontsize=8, fontweight="bold")
        fig.text(x, 0.795, value, color="#FFFFFF", fontsize=14 if i != 1 else 11,
                 fontweight="bold")

    left_x, right_x = 0.065, 0.525
    fig.text(left_x, 0.72, "DADOS E SÉRIE", color=TEAL, fontsize=10, fontweight="bold")
    fig.text(right_x, 0.72, "REDE E TREINAMENTO", color=CORAL, fontsize=10, fontweight="bold")

    data_lines = [
        ("FONTE", "UCI ML Repository · conjunto 235"),
        ("VOLUME DE ORIGEM", f"{config.UCI_TOTAL_ROWS:,} registros".replace(",", ".")),
        ("RECORTE USADO", f"{sample_count:,} registros limpos ({total_share:.1f}% do total)".replace(",", ".")),
        ("ALVO", "Energia acumulada no intervalo futuro (kWh)"),
        ("ENTRADAS", "7 medidas: potência ativa/reativa, tensão, corrente e submedições"),
        ("JANELAS", f"{config.LOOKBACK_MINUTES} min anteriores · amostragem a cada {config.WINDOW_STRIDE_MINUTES} min"),
        ("PREVISÕES", "Energia nos próximos 15, 30 e 60 minutos"),
        ("PADRONIZAÇÃO", "Features e alvos: StandardScaler ajustado só no treino"),
        ("LIMPEZA", "Marcadores ausentes interpolados pela série temporal"),
    ]
    y = 0.68
    for label, value in data_lines:
        fig.text(left_x, y, label, color=MUTED, fontsize=8, fontweight="bold")
        fig.text(left_x + 0.16, y, value, color=INK, fontsize=9)
        y -= 0.041

    feature_labels = [
        "Potência ativa e reativa (kW)",
        "Tensão (V) e corrente (A)",
        "Submedições 1, 2 e 3 (Wh)",
    ]
    fig.text(left_x, 0.335, "VARIÁVEIS DE ENTRADA", color=MUTED, fontsize=8, fontweight="bold")
    for i, label in enumerate(feature_labels):
        fig.text(left_x + 0.02, 0.30 - i * 0.033, f"•  {label}", color=INK, fontsize=9)

    model_lines = [
        ("ARQUITETURA", f"LSTM empilhada {config.LSTM_UNITS[0]} → {config.LSTM_UNITS[1]} · Dense {len(config.HORIZONS)} saídas"),
        ("FORMA / CADÊNCIA", f"{config.LOOKBACK_MINUTES} × {len(config.FEATURE_COLUMNS)} · janela a cada {config.WINDOW_STRIDE_MINUTES} min"),
        ("HORIZONTES", "15 min · 30 min · 1 h"),
        ("OTIMIZADOR", f"Adam · learning rate {config.LEARNING_RATE:g}"),
        ("OBJETIVO / MÉTRICA", "MSE / MAE"),
        ("REGULARIZAÇÃO", f"Dropout {config.DROPOUT_RATE:.0%}"),
        ("LOTE / ÉPOCAS", f"{config.BATCH_SIZE} / {config.EPOCHS}"),
        ("EARLY STOPPING", "val_loss · paciência 5 · restaurar melhores pesos"),
        ("REDUÇÃO DO LR", "fator 0,5 · paciência 3 · mínimo 0,00001"),
        ("REPRODUTIBILIDADE", f"semente aleatória {config.RANDOM_SEED}"),
    ]
    y = 0.68
    for label, value in model_lines:
        fig.text(right_x, y, label, color=MUTED, fontsize=8, fontweight="bold")
        fig.text(right_x + 0.17, y, value, color=INK, fontsize=9)
        y -= 0.041

    fig.text(right_x, 0.255, "CALLBACKS", color=MUTED, fontsize=8, fontweight="bold")
    fig.text(right_x + 0.17, 0.255, "melhor checkpoint por val_loss · ReduceLROnPlateau",
             color=INK, fontsize=8.5)

    table_ax = fig.add_axes([0.065, 0.075, 0.87, 0.125])
    table_ax.axis("off")
    split_table = table_ax.table(
        cellText=[[name, f"{count:,}".replace(",", "."),
                   start.strftime("%d/%m/%Y %H:%M"), end.strftime("%d/%m/%Y %H:%M")]
                  for name, start, end, count in split_rows],
        colLabels=["PARTIÇÃO", "AMOSTRAS", "INÍCIO", "FIM"],
        cellLoc="center", colLoc="center", colWidths=[0.18, 0.18, 0.32, 0.32],
        bbox=[0, 0, 1, 1],
    )
    split_table.auto_set_font_size(False)
    split_table.set_fontsize(8.5)
    for (row, col), cell in split_table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="#FFFFFF", weight="bold")
        else:
            cell.set_facecolor("#E9EFEB" if row % 2 else "#FFFFFF")
            cell.set_text_props(color=INK, weight="bold" if col == 0 else "normal")

    fig.savefig(config.FIGURES_DIR / "experiment_summary.png", dpi=220,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: dict[str, int] = config.HORIZONS, n_points: int = 500
) -> None:
    """Time-series overlay of actual vs predicted values for a sample window of the test set."""
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.25 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]
    _style_figure(fig, "Consumo previsto", "Energia ativa acumulada no intervalo futuro | conjunto de teste | kWh")
    for i, name in enumerate(horizons):
        ax = axes[i]
        _style_axis(ax)
        color = HORIZON_COLORS[i % len(HORIZON_COLORS)]
        x = np.arange(min(n_points, len(y_true)))
        ax.fill_between(x, y_true[:n_points, i], y_pred[:n_points, i], color=color, alpha=0.10)
        ax.plot(x, y_true[:n_points, i], color=INK, label="Real", linewidth=1.45)
        ax.plot(x, y_pred[:n_points, i], color=color, label="LSTM", linewidth=1.5, alpha=0.95)
        ax.set_title(name.upper(), loc="left", color=INK, fontsize=11, fontweight="bold", pad=9)
        ax.set_ylabel("Energia no intervalo (kWh)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, ncol=2, loc="upper right", labelcolor=INK, fontsize=9)
    axes[-1].set_xlabel("Janela de previsão no conjunto de teste", color=MUTED, fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.885, bottom=0.07, hspace=0.4)
    fig.savefig(config.FIGURES_DIR / "predictions_vs_actual.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scatter_and_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: dict[str, int] = config.HORIZONS
) -> None:
    fig, axes = plt.subplots(2, len(horizons), figsize=(5.2 * len(horizons), 7.2), squeeze=False)
    _style_figure(fig, "Diagnóstico dos erros", "Conjunto de teste | linha diagonal representa previsão perfeita")
    for i, name in enumerate(horizons):
        color = HORIZON_COLORS[i % len(HORIZON_COLORS)]
        true_i, pred_i = y_true[:, i], y_pred[:, i]
        residuals = true_i - pred_i
        scatter_ax, residual_ax = axes[0, i], axes[1, i]
        _style_axis(scatter_ax)
        _style_axis(residual_ax)

        stride = max(1, len(true_i) // 6000)
        scatter_ax.scatter(true_i[::stride], pred_i[::stride], s=9, alpha=0.24,
                           color=color, edgecolors="none", rasterized=True)
        lims = [min(true_i.min(), pred_i.min()), max(true_i.max(), pred_i.max())]
        scatter_ax.plot(lims, lims, color=INK, linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.75)
        scatter_ax.set_title(f"REAL × LSTM · {name.upper()}", loc="left", color=INK,
                             fontsize=10, fontweight="bold", pad=9)
        scatter_ax.set_xlabel("Real (kWh)", color=MUTED, fontsize=9)
        scatter_ax.set_ylabel("Previsto (kWh)", color=MUTED, fontsize=9)

        residual_ax.hist(residuals, bins=45, color=color, edgecolor=PAPER, linewidth=0.45)
        residual_ax.axvline(0, color=INK, linestyle=(0, (4, 3)), linewidth=1.1)
        residual_ax.set_title(f"RESÍDUOS · {name.upper()}", loc="left", color=INK,
                             fontsize=10, fontweight="bold", pad=9)
        residual_ax.set_xlabel("Real − previsto (kWh)", color=MUTED, fontsize=9)
        residual_ax.set_ylabel("Frequência", color=MUTED, fontsize=9)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.85, bottom=0.09, wspace=0.3, hspace=0.36)
    fig.savefig(config.FIGURES_DIR / "scatter_residuals.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    target_center: np.ndarray | float = 0.0,
    target_scale: np.ndarray | float = 1.0,
) -> pd.DataFrame:
    """Evaluate predicted interval energy and report metrics in kWh."""
    y_true, y_pred = collect_predictions(model, test_dataset)
    y_true = y_true * target_scale + target_center
    y_pred = y_pred * target_scale + target_center

    metrics_df = compute_metrics(y_true, y_pred)
    save_metrics(metrics_df)

    plot_experiment_summary(pd.read_parquet(config.CLEAN_PARQUET_PATH))
    plot_metrics_table(metrics_df)
    plot_predictions_vs_actual(y_true, y_pred)
    plot_scatter_and_residuals(y_true, y_pred)

    return metrics_df
