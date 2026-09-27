"""Model preprocessing must be learned exclusively from training observations."""
import unittest

import numpy as np

from src.experiments import prepare_neural, score


class TransformTests(unittest.TestCase):
    def test_heldout_extremes_cannot_change_fitted_scaler_or_pca(self):
        rng = np.random.default_rng(9)
        train = {"sequence": rng.normal(size=(30, 24, 7)).astype("float32"),
                 "context": rng.normal(size=(30, 4)).astype("float32"),
                 "y": rng.uniform(.1, 3, (30, 3)).astype("float32")}
        holdout = {key: values.copy() for key, values in train.items()}
        _, first = prepare_neural(train, [holdout], True, True)
        holdout["sequence"] += 10000
        holdout["context"] *= 500
        holdout["y"] += 10000
        _, second = prepare_neural(train, [holdout], True, True)
        np.testing.assert_array_equal(first["sequence_scaler"].mean_, second["sequence_scaler"].mean_)
        np.testing.assert_array_equal(first["target_scaler"].mean_, second["target_scaler"].mean_)
        np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)
        np.testing.assert_array_equal(first["context_transform"][-1].mean_, second["context_transform"][-1].mean_)

    def test_equal_power_errors_have_equal_weight_across_horizons(self):
        self.assertAlmostEqual(score(np.zeros((2, 3)), np.array([[.25, .5, 1]]*2)), 1.)


if __name__ == "__main__":
    unittest.main()
