"""Model training with early stopping/checkpointing and a saved loss-curve plot."""
from __future__ import annotations

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from . import config


def get_callbacks(model_name: str = "lstm_energy_forecaster") -> list[keras.callbacks.Callback]:
    checkpoint_path = config.MODELS_DIR / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = config.EPOCHS,
) -> keras.callbacks.History:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=get_callbacks(),
        shuffle=False,
        verbose=2,
    )
    plot_training_history(history)
    return history


def plot_training_history(history: keras.callbacks.History) -> None:
    ink, muted, paper, grid = "#18332F", "#64736F", "#F7F6F1", "#DDE2DB"
    teal, coral = "#087F78", "#D86B50"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor(paper)
    fig.suptitle("Aprendizagem da LSTM", x=0.075, y=0.99, ha="left", color=ink,
                 fontsize=19, fontweight="bold", fontfamily="Georgia")
    fig.text(0.075, 0.90, "MSE e MAE dos alvos de energia padronizados · treino e validação",
             ha="left", color=muted, fontsize=10)

    panels = [("loss", "val_loss", "Perda quadrática média", "MSE", teal),
              ("mae", "val_mae", "Erro absoluto médio", "MAE", coral)]
    for ax, (train_key, val_key, title, ylabel, color) in zip(axes, panels):
        ax.set_facecolor(paper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        train_values = history.history[train_key]
        val_values = history.history[val_key]
        epochs = range(1, len(train_values) + 1)
        ax.plot(epochs, train_values, color=ink, linewidth=1.5, label="Treino")
        ax.plot(epochs, val_values, color=color, linewidth=1.8, label="Validação")
        best_epoch = int(min(range(len(val_values)), key=val_values.__getitem__))
        ax.scatter(best_epoch + 1, val_values[best_epoch], color=color, s=32, zorder=3)
        ax.annotate(f"mínimo {val_values[best_epoch]:.3f}",
                    (best_epoch + 1, val_values[best_epoch]), xytext=(7, 10),
                    textcoords="offset points", color=color, fontsize=8)
        ax.set_title(title, loc="left", color=ink, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Época", color=muted, fontsize=9)
        ax.set_ylabel(ylabel, color=muted, fontsize=9)
        ax.legend(frameon=False, labelcolor=ink, fontsize=9)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.15, wspace=0.24)
    fig.savefig(config.FIGURES_DIR / "training_history.png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
