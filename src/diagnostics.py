"""Dataset coverage and PCA reports, with transforms fitted on training only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config


def pca_report(frame: pd.DataFrame, name: str, output: Path,
               complete_cases: bool = False) -> dict:
    fit_frame = frame.dropna() if complete_cases else frame
    imputed = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(fit_frame)
    z = StandardScaler().fit_transform(imputed)
    pca = PCA(svd_solver="full").fit(z)
    ratios = pca.explained_variance_ratio_
    cumulative = ratios.cumsum()
    summary = {f"components_{int(level*100)}pct": int(np.searchsorted(cumulative, level) + 1)
               for level in (.90, .95, .99)}
    summary.update(features=len(frame.columns), samples=len(fit_frame),
                   explained_variance_ratio=ratios.tolist())
    pd.DataFrame({"component": np.arange(1, len(ratios)+1),
                  "explained_variance_ratio": ratios,
                  "cumulative_variance": cumulative}).to_csv(output / f"pca_{name}.csv", index=False)
    pd.DataFrame(pca.components_.T, index=frame.columns,
                 columns=[f"PC{i+1}" for i in range(len(ratios))]).to_csv(output / f"pca_{name}_weights.csv")
    frame.describe().T.assign(variance=frame.var(), missing_fraction=frame.isna().mean()).to_csv(
        output / f"feature_statistics_{name}.csv")
    frame.corr().to_csv(output / f"correlation_{name}.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(1, len(ratios)+1)
    ax.bar(x, ratios, color="#087F78", alpha=.6, label="Individual")
    ax.plot(x, cumulative, "o-", color="#D86B50", markersize=3, label="Acumulada")
    ax.axhline(.95, linestyle="--", color="#64736F", label="95%")
    ax.set(xlabel="Componente principal", ylabel="Fração da variância",
           title=f"PCA — {name} — ajustado somente no treino", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"pca_{name}.png", dpi=180)
    plt.close(fig)
    return summary


def audit_dataset(df: pd.DataFrame, train_end: int, test_start: int,
                  recent_start: int, output: Path) -> dict:
    missing = df[config.FEATURE_COLUMNS].isna().any(axis=1)
    groups = missing.ne(missing.shift()).cumsum()
    gap_lengths = missing[missing].groupby(groups[missing]).size()
    train = df.iloc[:train_end]
    recent = df.iloc[recent_start:train_end]
    summary = {
        "total_rows": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
        "missing_rows": int(missing.sum()), "missing_fraction": float(missing.mean()),
        "longest_gap_minutes": int(gap_lengths.max()) if len(gap_lengths) else 0,
        "original_train_rows": len(recent),
        "original_train_missing_fraction": float(missing.iloc[recent_start:train_end].mean()),
        "full_train_days": float((df.index[train_end-1] - df.index[0]).total_seconds()/86400),
        "original_train_days": len(recent)/1440,
        "power_current_correlation": float(train.Global_active_power.corr(train.Global_intensity)),
        "partitions": {},
    }
    for name, part in (("train", train), ("validation", df.iloc[train_end:test_start]),
                        ("test", df.iloc[test_start:])):
        summary["partitions"][name] = {"rows": len(part), "start": str(part.index[0]),
                                     "end": str(part.index[-1])}
    monthly = df[config.FEATURE_COLUMNS].resample("MS").agg(["mean", "std", "count"])
    monthly.to_csv(output / "monthly_coverage.csv")
    pd.DataFrame({"rows": missing.resample("MS").size(),
                  "missing_rows": missing.resample("MS").sum()}).to_csv(output / "monthly_missing.csv")
    summary["pca_original_train"] = pca_report(recent, "original_train", output, complete_cases=True)
    summary["pca_full_train"] = pca_report(train, "full_train", output, complete_cases=True)
    with (output / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def plot_learning_curve(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(table["samples"], table["train_score"], "o-", label="Treino")
    ax.plot(table["samples"], table["validation_score"], "o-", label="Validação fixa")
    ax.set(xlabel="Janelas de treino (histórico crescente)", ylabel="RMSE médio equivalente em kW",
           title="Volume e cobertura temporal — HistGradientBoosting")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=180)
    plt.close(fig)
