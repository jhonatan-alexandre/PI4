"""Bounded two-fold temporal search with checkpoint selection in physical kWh."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from tensorflow import keras
from threadpoolctl import threadpool_limits

from . import config, data_loader, features, preprocessing
from .experiments import file_hash, metrics, prepare_neural, predict_neural


TREE_GRID = [
    dict(name="hgb_reference", max_iter=200, max_leaf_nodes=15, learning_rate=.08,
         min_samples_leaf=40, l2_regularization=5., loss="squared_error", enhanced=False),
    dict(name="hgb_31", max_iter=400, max_leaf_nodes=31, learning_rate=.05,
         min_samples_leaf=60, l2_regularization=10., loss="squared_error", enhanced=True),
    dict(name="hgb_regularized", max_iter=600, max_leaf_nodes=31, learning_rate=.04,
         min_samples_leaf=100, l2_regularization=20., loss="squared_error", enhanced=True),
    dict(name="hgb_63", max_iter=400, max_leaf_nodes=63, learning_rate=.05,
         min_samples_leaf=100, l2_regularization=20., loss="squared_error", enhanced=True),
    dict(name="hgb_mae", max_iter=300, max_leaf_nodes=31, learning_rate=.06,
         min_samples_leaf=100, l2_regularization=10., loss="absolute_error", enhanced=True),
]
NEURAL_GRID = [
    dict(name="lstm_full_mse", units=32, dropout=.10, learning_rate=.001, loss="mse"),
    dict(name="lstm_full_huber", units=64, dropout=.15, learning_rate=.001, loss="huber"),
    dict(name="lstm_full_mixed", units=32, dropout=.10, learning_rate=.0007, loss="mixed"),
]


def validate_resume_state(manifest: dict, current_code: dict, current_versions: dict) -> None:
    for name, digest in current_code.items():
        if manifest.get("code_sha256", {}).get(name) != digest:
            raise ValueError(f"Resume code changed: {name}. Start a new output directory.")
    for name, version in current_versions.items():
        if manifest.get("versions", {}).get(name) != version:
            raise ValueError(f"Resume library version changed: {name}.")


def kwh_score(y: np.ndarray, prediction: np.ndarray) -> float:
    """Mean of MAE and RMSE across horizons, both in kWh, lower is better."""
    error = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return float(.5 * (np.mean(np.abs(error), axis=0) +
                       np.sqrt(np.mean(error**2, axis=0))).mean())


def physical(prediction: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.maximum(prediction, 0), axis=1)


def training_mask(origins: np.ndarray, end: int, stride: int) -> np.ndarray:
    return (origins + 60 <= end) & (origins >= 120) & ((origins - 120) % stride == 0)


def tree_inputs(samples: dict, enhanced: bool) -> np.ndarray:
    if not enhanced:
        return samples["context"]
    seq = samples["sequence"]
    # Recent power trajectory and per-channel trend/range help identify state changes.
    return np.column_stack([samples["context"], seq[:, :, 0], seq[:, -1]-seq[:, 0],
                            seq[:, -3:].mean(axis=1), seq.max(axis=1)-seq.min(axis=1)])


@keras.utils.register_keras_serializable(package="energy")
class KwhLoss(keras.losses.Loss):
    def __init__(self, scale, mode="mse", **kwargs):
        super().__init__(**kwargs)
        self.scale = list(scale)
        self.mode = mode

    def call(self, y_true, y_pred):
        error = (y_pred-y_true) * tf.cast(self.scale, y_pred.dtype)
        magnitude = tf.abs(error)
        if self.mode == "huber":
            loss = tf.where(magnitude <= .2, .5*error**2, .2*(magnitude-.1))
        elif self.mode == "mixed":
            loss = .5*magnitude + .5*error**2
        else:
            loss = error**2
        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        return {**super().get_config(), "scale": self.scale, "mode": self.mode}


def neural_inputs(samples: dict, transforms: dict) -> dict:
    seq = samples["sequence"]
    return {"sequence": transforms["sequence_scaler"].transform(
                seq.reshape(-1, 7)).reshape(seq.shape).astype("float32"),
            "context": transforms["context_transform"].transform(samples["context"]).astype("float32")}


def infer_arrays(model, inputs: dict, scaler) -> np.ndarray:
    result = np.concatenate([model({k: v[i:i+512] for k, v in inputs.items()}, training=False).numpy()
                             for i in range(0, len(inputs["sequence"]), 512)])
    return physical(scaler.inverse_transform(result))


class KwhValidation(keras.callbacks.Callback):
    def __init__(self, inputs: dict, truth: np.ndarray, scaler):
        super().__init__()
        self.inputs, self.truth, self.scaler = inputs, truth, scaler
        self.best_epoch = 0
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        value = kwh_score(self.truth, infer_arrays(self.model, self.inputs, self.scaler))
        logs["val_kwh_score"] = value
        if value < self.best:
            self.best, self.best_epoch = value, epoch+1


def fit_network(train: dict, val: dict | None, params: dict, epochs: int,
                path: Path, learning_rates: list[float] | None = None) -> tuple[dict, np.ndarray | None]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    arrays, transforms = prepare_neural(train, [] if val is None else [val], True, False)
    seq = keras.Input(shape=arrays[0]["sequence"].shape[1:], name="sequence")
    ctx = keras.Input(shape=(arrays[0]["context"].shape[1],), name="context")
    hidden = keras.layers.LSTM(params["units"])(seq)
    hidden = keras.layers.Concatenate()([hidden, ctx])
    hidden = keras.layers.Dense(96, activation="relu")(hidden)
    hidden = keras.layers.Dropout(params["dropout"])(hidden)
    hidden = keras.layers.Dense(48, activation="relu")(hidden)
    output = keras.layers.Dense(3)(hidden)
    model = keras.Model({"sequence": seq, "context": ctx}, output)
    scaler = transforms["target_scaler"]
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"], clipnorm=1.),
                  loss=KwhLoss(scaler.scale_.tolist(), params["loss"]))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    ds = tf.data.Dataset.from_tensor_slices((arrays[0], scaler.transform(train["y"]).astype("float32")))
    ds = ds.shuffle(len(train["y"]), seed=42).batch(256).with_options(options).prefetch(1)
    monitor = None
    callbacks = []
    if learning_rates is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda epoch, current: learning_rates[min(epoch, len(learning_rates)-1)]))
    if val is not None:
        monitor = KwhValidation(arrays[1], val["y"], scaler)
        callbacks = [monitor, keras.callbacks.EarlyStopping(monitor="val_kwh_score", mode="min",
                        patience=6, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor="val_kwh_score", mode="min",
                        patience=3, factor=.5, min_lr=1e-5)]
    history = model.fit(ds, epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False)
    model.save(path.with_suffix(".keras"))
    joblib.dump(transforms, path.parent / f"{path.name}_transforms.joblib")
    pd.DataFrame(history.history).to_csv(path.parent / f"{path.name}_history.csv", index=False)
    pred = None if val is None else infer_arrays(model, arrays[1], scaler)
    info = {"best_epoch": epochs if monitor is None else monitor.best_epoch,
            "parameter_count": int(model.count_params())}
    return info, pred


def fit_tree(train: dict, params: dict):
    args = {k: v for k, v in params.items() if k not in ("name", "enhanced")}
    estimator = MultiOutputRegressor(HistGradientBoostingRegressor(
        **args, early_stopping=False, random_state=42))
    with threadpool_limits(limits=4):
        estimator.fit(tree_inputs(train, params["enhanced"]), train["y"])
    return estimator


def predict_tree(estimator, samples: dict, params: dict) -> np.ndarray:
    with threadpool_limits(limits=4):
        return physical(estimator.predict(tree_inputs(samples, params["enhanced"])))


def predict_selected(output: Path, samples: dict) -> np.ndarray:
    """Load the final selected model/ensemble and return interval energy in kWh."""
    final = output / "final"
    setup = json.loads((final / "model_config.json").read_text(encoding="utf-8"))
    if samples["context_columns"] != setup["context_columns"]:
        raise ValueError("Context feature names/order do not match the trained model.")
    winner = setup["winner"]
    tree_prediction = neural_prediction = None
    if winner in (setup["best_tree"], "ensemble"):
        tree_prediction = predict_tree(joblib.load(final / f"{setup['best_tree']}.joblib"),
                                       samples, setup["tree_parameters"])
    if winner in (setup["best_neural"], "ensemble"):
        network = keras.models.load_model(final / f"{setup['best_neural']}.keras", compile=False)
        transforms = joblib.load(final / f"{setup['best_neural']}_transforms.joblib")
        neural_prediction = predict_neural(network, transforms, samples)
    if winner == "ensemble":
        return setup["tree_weight"]*tree_prediction + (1-setup["tree_weight"])*neural_prediction
    if tree_prediction is not None:
        return tree_prediction
    if neural_prediction is not None:
        return neural_prediction
    raise ValueError(f"Unknown selected model: {winner}")


def run_search(output: Path, reference: Path, epochs: int = 35,
               stride: int = 30, resume: bool = False) -> None:
    if epochs < 1 or stride < 1:
        raise ValueError("epochs and stride must be positive.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a new output directory or --resume.")
    output.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    signature = {"epochs": epochs, "stride": stride, "seed": 42,
                 "reference": str(reference.resolve()), "trees": TREE_GRID, "networks": NEURAL_GRID}
    protocol_files = ["tune.py", "src/tuning.py", "src/features.py", "src/experiments.py",
                      "src/config.py", "src/preprocessing.py", "src/data_loader.py"]
    current_code = {str(Path(name)): file_hash(config.ROOT_DIR / name) for name in protocol_files}
    current_versions = {"tensorflow": tf.__version__, "numpy": np.__version__,
                        "sklearn": __import__("sklearn").__version__, "pandas": pd.__version__}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["search"] != signature or manifest["raw_sha256"] != file_hash(config.RAW_CSV_PATH):
            raise ValueError("Resume configuration or input data differs from the original run.")
        validate_resume_state(manifest, current_code, current_versions)
    else:
        ref = output / "reference"
        ref.mkdir()
        names = ["test_predictions.csv", "test_metrics.csv", "hgb_full.joblib",
                 "lstm_features.keras", "lstm_features_transforms.joblib"]
        for name in names:
            shutil.copy2(reference/name, ref/name)
        manifest = {"search": signature, "objective": "mean_horizons(0.5*MAE_kWh + 0.5*RMSE_kWh)",
                    "test_status": "previously inspected historical benchmark; not a fresh holdout",
                    "raw_sha256": file_hash(config.RAW_CSV_PATH),
                    "reference_sha256": {name: file_hash(ref/name) for name in names},
                    "code_sha256": {str(p.relative_to(config.ROOT_DIR)): file_hash(p)
                                    for p in [config.ROOT_DIR / "tune.py"]+
                                    sorted((config.ROOT_DIR / "src").glob("*.py"))},
                    "versions": current_versions}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cache = output / "prepared_samples.joblib"
    if cache.exists():
        samples, boundaries = joblib.load(cache)
    else:
        print("Preparing observed-only causal samples from full history...", flush=True)
        df = preprocessing.clean_numeric_columns(preprocessing.parse_datetime_index(data_loader.load_raw_data()))
        end = len(df)
        boundaries = {"fold1": (end-90000, end-60000), "fold2": (end-60000, end-30000),
                      "test": (end-30000, end)}
        origins = [features.partition_origins(0, end-30000, 120, stride)]
        origins += [features.partition_origins(start, stop, 120, 5) for start, stop in boundaries.values()]
        samples = features.make_samples(df, np.unique(np.concatenate(origins)))
        joblib.dump((samples, boundaries), cache, compress=1)
        del df
    series_start = samples["times"][0] - pd.Timedelta(minutes=int(samples["origins"][0]))
    manifest["boundaries"] = {
        name: {"start": str(series_start + pd.Timedelta(minutes=start)),
               "stop_exclusive": str(series_start + pd.Timedelta(minutes=stop)),
               "start_row": start, "stop_row_exclusive": stop,
               "first_forecast": str(features.subset(samples, np.isin(samples["origins"],
                   features.partition_origins(start, stop, 120, 5)))["times"][0])}
        for name, (start, stop) in boundaries.items()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    folds = {}
    for name in ("fold1", "fold2"):
        start, stop = boundaries[name]
        val_origins = features.partition_origins(start, stop, 120, 5)
        folds[name] = (features.subset(samples, training_mask(samples["origins"], start, stride)),
                       features.subset(samples, np.isin(samples["origins"], val_origins)))
    records, predictions = [], {}
    for params in TREE_GRID + NEURAL_GRID:
        name = params["name"]
        predictions[name] = {}
        for fold, (train, val) in folds.items():
            folder = output / fold
            folder.mkdir(exist_ok=True)
            prefix = folder / name
            done = folder / f"{name}_result.json"
            if done.exists():
                result = json.loads(done.read_text(encoding="utf-8"))
                pred = np.load(folder / f"{name}_validation.npy")
            else:
                print(f"SEARCH {name} {fold}: train={len(train['y'])}, val={len(val['y'])}", flush=True)
                started = time.perf_counter()
                info = {}
                if name.startswith("hgb"):
                    estimator = fit_tree(train, params)
                    joblib.dump(estimator, prefix.with_suffix(".joblib"))
                    pred = predict_tree(estimator, val, params)
                else:
                    info, pred = fit_network(train, val, params, epochs, prefix)
                result = {"model": name, "fold": fold, "score_kwh": kwh_score(val["y"], pred),
                          "seconds": time.perf_counter()-started, "parameters": params, **info}
                np.save(folder / f"{name}_validation.npy", pred)
                metrics(val["y"], pred, name, fold).to_csv(folder / f"{name}_metrics.csv", index=False)
                done.write_text(json.dumps(result, indent=2), encoding="utf-8")
            predictions[name][fold] = pred
            records.append(result)
            print(f"RESULT {name} {fold}: {result['score_kwh']:.6f} kWh", flush=True)
            pd.DataFrame(records).to_csv(output / "search_trials.csv", index=False)
    average = {name: float(np.mean([kwh_score(val["y"], predictions[name][fold])
                                   for fold, (_, val) in folds.items()])) for name in predictions}
    best_tree = min((p["name"] for p in TREE_GRID), key=average.get)
    best_neural = min((p["name"] for p in NEURAL_GRID), key=average.get)
    blends = {}
    for weight in (.25, .5, .75):
        blends[str(weight)] = float(np.mean([kwh_score(val["y"],
            weight*predictions[best_tree][fold]+(1-weight)*predictions[best_neural][fold])
            for fold, (_, val) in folds.items()]))
    best_weight = min(blends, key=blends.get)
    winner = min(average, key=average.get)
    ensemble_weight = None
    if blends[best_weight] < average[winner]:
        winner, ensemble_weight = "ensemble", float(best_weight)
    selection = {"winner": winner, "best_tree": best_tree, "best_neural": best_neural,
                 "tree_weight": ensemble_weight, "mean_validation_scores": average, "blend_scores": blends,
                 "protocol": "two chronological folds, selection before test; refit including validation"}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print("SELECTION FROZEN", selection, flush=True)
    final_train = features.subset(samples, training_mask(samples["origins"], boundaries["test"][0], stride))
    test_origins = features.partition_origins(*boundaries["test"], 120, 5)
    test = features.subset(samples, np.isin(samples["origins"], test_origins))
    final = output / "final"
    final.mkdir(exist_ok=True)
    test_predictions = {}
    for name in (best_tree, best_neural):
        params = next(p for p in TREE_GRID+NEURAL_GRID if p["name"] == name)
        print(f"FINAL REFIT {name}: {len(final_train['y'])} windows", flush=True)
        if name.startswith("hgb"):
            estimator = fit_tree(final_train, params)
            joblib.dump(estimator, final / f"{name}.joblib")
            test_predictions[name] = predict_tree(estimator, test, params)
        else:
            fixed_epochs = max(1, round(np.mean([r["best_epoch"] for r in records if r["model"] == name])))
            fold_history = pd.read_csv(output / "fold2" / f"{name}_history.csv")
            rates = fold_history["learning_rate"].tolist()
            rates = [rates[min(i, len(rates)-1)] for i in range(fixed_epochs)]
            fit_network(final_train, None, params, fixed_epochs, final/name, learning_rates=rates)
            selection["final_neural_epochs"] = fixed_epochs
            selection["final_neural_learning_rates"] = rates
            net = keras.models.load_model(final / f"{name}.keras", compile=False)
            transform = joblib.load(final / f"{name}_transforms.joblib")
            test_predictions[name] = predict_neural(net, transform, test)
    if ensemble_weight is not None:
        test_predictions["ensemble"] = (ensemble_weight*test_predictions[best_tree]+
                                       (1-ensemble_weight)*test_predictions[best_neural])
    # Historical predictions are read only after model/ensemble selection is frozen.
    reference_predictions = pd.read_csv(output / "reference" / "test_predictions.csv", parse_dates=["time"])
    matched = reference_predictions.set_index("time").loc[test["times"]]
    for name in ("lstm_original_saved", "hgb_full", "lstm_features"):
        test_predictions[f"previous_{name}"] = matched[[f"{name}_{h}" for h in config.HORIZONS]].to_numpy()
    np.testing.assert_allclose(test["y"], matched[[f"actual_{h}" for h in config.HORIZONS]], rtol=1e-5, atol=1e-6)
    table = pd.concat([metrics(test["y"], pred, name, "historical_test")
                       for name, pred in test_predictions.items()], ignore_index=True)
    table.to_csv(output / "test_metrics.csv", index=False)
    saved = pd.DataFrame({"time": test["times"]})
    for j, horizon in enumerate(config.HORIZONS):
        saved[f"actual_{horizon}"] = test["y"][:, j]
        for name, pred in test_predictions.items():
            saved[f"{name}_{horizon}"] = pred[:, j]
    saved.to_csv(output / "test_predictions.csv", index=False)
    (final / "model_config.json").write_text(json.dumps({**selection,
        "context_columns": samples["context_columns"], "lookback_minutes": 120,
        "bin_minutes": 5, "train_stride": stride, "train_samples": len(final_train["y"]),
        "tree_parameters": next(p for p in TREE_GRID if p["name"] == best_tree),
        "neural_parameters": next(p for p in NEURAL_GRID if p["name"] == best_neural)}, indent=2), encoding="utf-8")
    from .tuning_report import generate_tuning_report
    generate_tuning_report(output)
    print(table.to_string(index=False), flush=True)
