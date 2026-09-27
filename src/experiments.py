"""Reproducible temporal experiments; selection uses validation, never test."""
from __future__ import annotations

import json
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from tensorflow import keras

from . import config, data_loader, diagnostics, features, preprocessing


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_legacy(output: Path, source: Path | None = None) -> dict:
    """Freeze the baseline before training so other runs cannot replace it."""
    source = config.MODELS_DIR if source is None else source
    filenames = ("lstm_energy_forecaster.keras", "feature_scaler.joblib", "energy_target_scaler.joblib")
    metadata = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source.resolve()), "artifacts": {}}
    absent = [name for name in filenames if not (source / name).exists()]
    if absent:
        metadata.update(status="unavailable", missing=absent)
        print(f"Legacy comparison unavailable; missing: {absent}", flush=True)
    else:
        destination = output / "legacy"
        destination.mkdir(exist_ok=False)
        for name in filenames:
            shutil.copy2(source / name, destination / name)
            artifact = {"sha256": file_hash(destination / name)}
            if name.endswith(".joblib"):
                scaler = joblib.load(destination / name)
                artifact.update(samples_seen=np.asarray(scaler.n_samples_seen_).tolist(),
                                features=int(scaler.n_features_in_))
            metadata["artifacts"][name] = artifact
        metadata["status"] = "available"
    (output / "legacy_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def metrics(y: np.ndarray, pred: np.ndarray, name: str, split: str) -> pd.DataFrame:
    rows = []
    for j, horizon in enumerate(config.HORIZONS):
        error = np.abs(y[:, j] - pred[:, j])
        rows.append({"model": name, "split": split, "horizonte": horizon,
                     "samples": len(y), "MAE": mean_absolute_error(y[:, j], pred[:, j]),
                     "RMSE": np.sqrt(mean_squared_error(y[:, j], pred[:, j])),
                     "MAPE_%": 100*np.mean(error / np.maximum(np.abs(y[:, j]), 1e-8)),
                     "WAPE_%": 100*error.sum()/np.abs(y[:, j]).sum(),
                     "R2": r2_score(y[:, j], pred[:, j])})
    return pd.DataFrame(rows)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    """Equal horizon weights in comparable units (interval-average power)."""
    hours = np.array(list(config.HORIZONS.values())) / 60
    return float(np.mean(np.sqrt(np.mean((y - pred)**2, axis=0)) / hours))


def persistence(samples: dict) -> np.ndarray:
    return samples["sequence"][:, -1, 0, None] * (np.array(list(config.HORIZONS.values()))/60)


def train_boosting(train: dict, validation: dict) -> tuple[object, np.ndarray, float]:
    estimator = MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=15, min_samples_leaf=40, learning_rate=.08,
        l2_regularization=5., early_stopping=False, random_state=config.RANDOM_SEED))
    with threadpool_limits(limits=4):
        estimator.fit(train["context"], train["y"])
        prediction = estimator.predict(validation["context"])
        train_score = score(train["y"], estimator.predict(train["context"]))
    return estimator, prediction, train_score


def prepare_neural(train: dict, others: list[dict], use_context: bool, use_pca: bool):
    sequence_scaler = StandardScaler().fit(train["sequence"].reshape(-1, 7))
    context_transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                    keep_empty_features=True), StandardScaler())
    context_transform.fit(train["context"])
    target_scaler = StandardScaler().fit(train["y"])
    pca = None
    if use_pca:
        pca = PCA(n_components=.95, svd_solver="full").fit(
            sequence_scaler.transform(train["sequence"].reshape(-1, 7)))

    def transform(samples):
        shape = samples["sequence"].shape
        seq = sequence_scaler.transform(samples["sequence"].reshape(-1, 7))
        if pca is not None:
            seq = pca.transform(seq)
        seq = seq.reshape(shape[0], shape[1], -1).astype("float32")
        result = {"sequence": seq}
        if use_context:
            result["context"] = context_transform.transform(samples["context"]).astype("float32")
        return result

    transforms = {"sequence_scaler": sequence_scaler, "context_transform": context_transform,
                  "target_scaler": target_scaler, "pca": pca, "use_context": use_context}
    return [transform(s) for s in [train] + others], transforms


def build_network(inputs: dict, use_context: bool) -> keras.Model:
    seq = keras.Input(shape=inputs["sequence"].shape[1:], name="sequence")
    encoded = keras.layers.LSTM(32)(seq)
    encoded = keras.layers.Dropout(.1)(encoded)
    model_inputs = {"sequence": seq}
    if use_context:
        ctx = keras.Input(shape=(inputs["context"].shape[1],), name="context")
        model_inputs["context"] = ctx
        encoded = keras.layers.Concatenate()([encoded, ctx])
    hidden = keras.layers.Dense(64, activation="relu")(encoded)
    hidden = keras.layers.Dropout(.1)(hidden)
    hidden = keras.layers.Dense(32, activation="relu")(hidden)
    output = keras.layers.Dense(len(config.HORIZONS))(hidden)
    model = keras.Model(model_inputs, output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.),
                  loss="mse", metrics=["mae"])
    return model


def train_neural(name: str, train: dict, val: dict, output: Path,
                 epochs: int, use_context: bool, use_pca: bool) -> tuple[object, dict, np.ndarray]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.RANDOM_SEED)
    arrays, transforms = prepare_neural(train, [val], use_context, use_pca)
    model = build_network(arrays[0], use_context)
    scaler = transforms["target_scaler"]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    def dataset(x, y, training):
        ds = tf.data.Dataset.from_tensor_slices((x, scaler.transform(y).astype("float32")))
        if training:
            ds = ds.shuffle(len(y), seed=config.RANDOM_SEED)
        return ds.batch(256).with_options(options).prefetch(1)
    started = time.perf_counter()
    history = model.fit(dataset(arrays[0], train["y"], True),
                        validation_data=dataset(arrays[1], val["y"], False), epochs=epochs,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                  restore_best_weights=True),
                                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5,
                                                                    patience=2, min_lr=1e-5)], shuffle=False, verbose=2)
    pd.DataFrame(history.history).to_csv(output / f"history_{name}.csv", index=False)
    model.save(output / f"{name}.keras")
    joblib.dump(transforms, output / f"{name}_transforms.joblib")
    pred = predict_neural(model, transforms, val)
    print(f"{name}: validation={score(val['y'], pred):.5f}, seconds={time.perf_counter()-started:.1f}", flush=True)
    return model, transforms, pred


def predict_neural(model, transforms: dict, samples: dict) -> np.ndarray:
    shape = samples["sequence"].shape
    seq = transforms["sequence_scaler"].transform(samples["sequence"].reshape(-1, 7))
    if transforms["pca"] is not None:
        seq = transforms["pca"].transform(seq)
    inputs = {"sequence": seq.reshape(shape[0], shape[1], -1).astype("float32")}
    if transforms["use_context"]:
        inputs["context"] = transforms["context_transform"].transform(samples["context"]).astype("float32")
    # Direct batches avoid a new tf.data thread pool for every prediction call.
    pred = np.concatenate([model({k: v[i:i+512] for k, v in inputs.items()}, training=False).numpy()
                           for i in range(0, shape[0], 512)])
    pred = transforms["target_scaler"].inverse_transform(pred)
    # Physical postprocessing is fixed in advance for every neural candidate.
    return np.maximum.accumulate(np.maximum(pred, 0), axis=1)


def predict_legacy(df: pd.DataFrame, samples: dict, model_directory: Path) -> np.ndarray | None:
    model_path = model_directory / "lstm_energy_forecaster.keras"
    scaler_path = model_directory / "feature_scaler.joblib"
    target_path = model_directory / "energy_target_scaler.joblib"
    if not all(p.exists() for p in (model_path, scaler_path, target_path)):
        return None
    old = keras.models.load_model(model_path, compile=False)
    scaler, target_scaler = joblib.load(scaler_path), joblib.load(target_path)
    lookback = old.input_shape[1]
    preds = []
    for i in range(0, len(samples["origins"]), 256):
        origins = samples["origins"][i:i+256]
        indexes = origins[:, None] + np.arange(-lookback, 0)
        values = df[config.FEATURE_COLUMNS].to_numpy("float32")[indexes]
        shape = values.shape
        frame = pd.DataFrame(values.reshape(-1, 7), columns=config.FEATURE_COLUMNS)
        x = scaler.transform(frame).reshape(shape).astype("float32")
        preds.append(old(x, training=False).numpy())
    return target_scaler.inverse_transform(np.concatenate(preds))


def run(output: Path, epochs: int = 25, train_stride: int = 15, neural_days: int = 365,
        reference_rows: int = 200000, lookback: int = 120, seed: int = 42) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if epochs <= 0 or train_stride <= 0 or neural_days <= 0 or reference_rows < 2000:
        raise ValueError("epochs, stride, days and reference_rows must be positive and sufficient.")
    output.mkdir(parents=True, exist_ok=True)
    legacy_provenance = snapshot_legacy(output)
    config.RANDOM_SEED = seed
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print("Loading complete raw series (no interpolated labels)...", flush=True)
    df = preprocessing.clean_numeric_columns(preprocessing.parse_datetime_index(data_loader.load_raw_data()))
    df = df[config.FEATURE_COLUMNS]
    if reference_rows > len(df):
        raise ValueError("reference_rows exceeds the raw dataset size.")
    recent_start = len(df) - reference_rows
    train_end = recent_start + int(reference_rows*config.TRAIN_FRAC)
    test_start = train_end + int(reference_rows*config.VAL_FRAC)
    audit = diagnostics.audit_dataset(df, train_end, test_start, recent_start, output)
    print("Building causal features and observed-only windows...", flush=True)
    train_origins = features.partition_origins(0, train_end, lookback, train_stride)
    val_origins = features.partition_origins(train_end, test_start, lookback, 5)
    test_origins = features.partition_origins(test_start, len(df), lookback, 5)
    all_samples = features.make_samples(df, np.r_[train_origins, val_origins, test_origins], lookback)
    train = features.subset(all_samples, all_samples["origins"] < train_end)
    val = features.subset(all_samples, (all_samples["origins"] >= train_end) &
                          (all_samples["origins"] < test_start))
    test = features.subset(all_samples, all_samples["origins"] >= test_start)
    del all_samples
    neural_train = features.subset(train, train["origins"] >= train_end - neural_days*1440)
    engineered = pd.DataFrame(neural_train["context"], columns=train["context_columns"])
    audit["pca_engineered_train"] = diagnostics.pca_report(engineered, "engineered_train", output)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"seed": seed, "epochs_limit": epochs, "lookback_minutes": lookback,
                "sequence_bin_minutes": 5, "train_stride": train_stride, "evaluation_stride": 5,
                "neural_days": neural_days, "reference_rows": reference_rows,
                "objective": "equal-horizon mean RMSE / horizon_hours, validation only",
                "context_columns": train["context_columns"],
                "partitions": audit["partitions"],
                "sample_counts": {"full_train": len(train["y"]), "neural_train": len(neural_train["y"]),
                                  "validation": len(val["y"]), "test": len(test["y"])},
                "versions": {"tensorflow": tf.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
                "raw_file_bytes": config.RAW_CSV_PATH.stat().st_size,
                "raw_file_sha256": file_hash(config.RAW_CSV_PATH),
                "legacy_provenance": legacy_provenance,
                "code_sha256": {str(path.relative_to(config.ROOT_DIR)): file_hash(path)
                                for path in [config.ROOT_DIR / "experiments.py"] +
                                sorted((config.ROOT_DIR / "src").glob("*.py"))}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print('SAMPLES', manifest['sample_counts'], flush=True)
    tables = [metrics(val["y"], persistence(val), "persistence_5min", "validation")]
    candidates = {"persistence_5min": {"score": score(val["y"], persistence(val)), "kind": "baseline"}}
    learning = []
    # Equal cadence/model/validation; only available historical coverage changes.
    for label, start in (("30days", train_end-30*1440), ("original", recent_start+lookback),
                         ("365days", train_end-365*1440), ("full", 0)):
        part = features.subset(train, train["origins"] >= max(lookback, start))
        print(f"Training HGB {label}: {len(part['y'])} windows...", flush=True)
        estimator, pred, train_score = train_boosting(part, val)
        name = f"hgb_{label}"
        joblib.dump(estimator, output / f"{name}.joblib")
        candidates[name] = {"score": score(val["y"], pred), "kind": "hgb"}
        learning.append({"history": label, "samples": len(part["y"]), "train_score": train_score,
                         "validation_score": candidates[name]["score"]})
        tables.append(metrics(val["y"], pred, name, "validation"))
        print(learning[-1], flush=True)
        pd.DataFrame(learning).to_csv(output / "learning_curve.csv", index=False)
    diagnostics.plot_learning_curve(pd.DataFrame(learning), output)
    for name, context, pca in (("lstm_raw", False, False),
                               ("lstm_features", True, False), ("lstm_features_pca", True, True)):
        print(f"Training {name}...", flush=True)
        net, transform, pred = train_neural(name, neural_train, val, output, epochs, context, pca)
        candidates[name] = {"score": score(val["y"], pred), "kind": "neural"}
        tables.append(metrics(val["y"], pred, name, "validation"))
        pd.concat(tables).to_csv(output / "validation_metrics.csv", index=False)
        del net, transform
    winner = min(candidates, key=lambda name: candidates[name]["score"])
    best_neural = min((name for name in candidates if candidates[name]["kind"] == "neural"),
                      key=lambda name: candidates[name]["score"])
    # Freeze selection before any test evaluation. Save it for auditability.
    selection = {"winner": winner, "best_neural": best_neural, "candidates": candidates}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selection frozen: {winner}; best neural: {best_neural}. Evaluating test once...", flush=True)
    predictions = {"persistence_5min": persistence(test)}
    for name in dict.fromkeys([winner, best_neural]):
        if candidates[name]["kind"] == "hgb":
            with threadpool_limits(limits=4):
                predictions[name] = joblib.load(output / f"{name}.joblib").predict(test["context"])
        elif candidates[name]["kind"] == "neural":
            net = keras.models.load_model(output / f"{name}.keras", compile=False)
            predictions[name] = predict_neural(net, joblib.load(output / f"{name}_transforms.joblib"), test)
    legacy = predict_legacy(df, test, output / "legacy")
    if legacy is not None:
        predictions["lstm_original_saved"] = legacy
    test_tables = [metrics(test["y"], pred, name, "test") for name, pred in predictions.items()]
    test_metrics = pd.concat(test_tables, ignore_index=True)
    test_metrics.to_csv(output / "test_metrics.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(output / "validation_metrics.csv", index=False)
    results = pd.DataFrame({"time": test["times"]})
    for j, horizon in enumerate(config.HORIZONS):
        results[f"actual_{horizon}"] = test["y"][:, j]
        for name, pred in predictions.items():
            results[f"{name}_{horizon}"] = pred[:, j]
    results.to_csv(output / "test_predictions.csv", index=False)
    from .reporting import generate_report
    generate_report(output)
    print(test_metrics.to_string(index=False), flush=True)
    print(f"Outputs: {output}", flush=True)
