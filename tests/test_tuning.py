"""The tuning objective and folds must preserve kWh and temporal boundaries."""
import unittest
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np


class TuningTests(unittest.TestCase):
    def test_objective_uses_kwh_without_dividing_by_horizon(self):
        from src.tuning import kwh_score
        self.assertAlmostEqual(kwh_score(np.zeros((2, 3)), np.array([[1., 2., 3.]]*2)), 2.)

    def test_training_labels_never_overlap_validation(self):
        from src.tuning import training_mask
        origins = np.array([120, 210, 240, 270, 300])
        np.testing.assert_array_equal(training_mask(origins, end=300, stride=30),
                                      [True, True, True, False, False])

    def test_augmented_features_use_only_existing_observed_sequence(self):
        from src.tuning import tree_inputs
        samples = {"context": np.array([[1., 2.]]),
                   "sequence": np.arange(168).reshape(1, 24, 7).astype(float)}
        out = tree_inputs(samples, enhanced=True)
        self.assertEqual(out.shape, (1, 2+24+21))
        np.testing.assert_array_equal(out[0, :2], [1, 2])
        np.testing.assert_array_equal(out[0, 2:26], np.arange(0, 168, 7))

    def test_resume_rejects_changed_training_code_and_library_versions(self):
        from src.tuning import validate_resume_state
        original = {"code_sha256": {"src/features.py": "original"},
                    "versions": {"tensorflow": "2.19.1"}}
        with self.assertRaisesRegex(ValueError, "code"):
            validate_resume_state(original, {"src/features.py": "changed"}, {"tensorflow": "2.19.1"})
        with self.assertRaisesRegex(ValueError, "version"):
            validate_resume_state(original, {"src/features.py": "original"}, {"tensorflow": "2.20.0"})
        validate_resume_state(original, {"src/features.py": "original"}, {"tensorflow": "2.19.1"})

    def test_loss_converts_scaled_residuals_to_kwh_and_serializes(self):
        from tensorflow import keras
        from src.tuning import KwhLoss
        truth = np.zeros((1, 3), dtype="float32")
        pred = np.array([[.5, .25, .125]], dtype="float32")
        for mode, expected in (("mse", 1.), ("mixed", 1.), ("huber", .18)):
            loss = KwhLoss([2., 4., 8.], mode)
            self.assertAlmostEqual(float(loss(truth, pred)), expected, places=6)
            restored = keras.losses.deserialize(keras.losses.serialize(loss))
            self.assertAlmostEqual(float(restored(truth, pred)), expected, places=6)

    def test_selected_artifact_returns_kwh_and_rejects_reordered_features(self):
        from sklearn.dummy import DummyRegressor
        from src.tuning import predict_selected
        samples = {"context_columns": ["a", "b"], "context": np.zeros((2, 2)),
                   "sequence": np.ones((2, 24, 7))}
        estimator = DummyRegressor(strategy="constant", constant=[.25, .5, 1.])
        estimator.fit(samples["context"], np.ones((2, 3)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final"
            final.mkdir()
            joblib.dump(estimator, final / "tree.joblib")
            setup = {"context_columns": ["a", "b"], "winner": "tree", "best_tree": "tree",
                     "best_neural": "network", "tree_parameters": {"enhanced": False}}
            (final / "model_config.json").write_text(json.dumps(setup), encoding="utf-8")
            np.testing.assert_allclose(predict_selected(root, samples), [[.25, .5, 1.]]*2)
            samples["context_columns"] = ["b", "a"]
            with self.assertRaisesRegex(ValueError, "feature"):
                predict_selected(root, samples)


if __name__ == "__main__":
    unittest.main()
