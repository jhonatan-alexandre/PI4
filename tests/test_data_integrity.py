"""Regression checks for temporal leakage, missing labels and dataset caching."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import config, preprocessing


def measurements(n=240):
    index = pd.date_range("2010-01-01", periods=n, freq="min")
    return pd.DataFrame({c: np.ones(n) for c in config.FEATURE_COLUMNS}, index=index)


class CacheTests(unittest.TestCase):
    def test_changing_row_limit_does_not_reuse_truncated_cache(self):
        frame = measurements(20)
        raw = frame.reset_index(drop=True)
        raw["Date"] = frame.index.strftime("%d/%m/%Y")
        raw["Time"] = frame.index.strftime("%H:%M:%S")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config, "CLEAN_PARQUET_PATH", Path(directory) / "clean.parquet"):
                with patch.object(config, "MAX_ROWS", 5):
                    self.assertEqual(len(preprocessing.build_clean_dataset(raw)), 5)
                with patch.object(config, "MAX_ROWS", 12):
                    self.assertEqual(len(preprocessing.build_clean_dataset(raw)), 12)
                with patch.object(config, "MAX_ROWS", None):
                    self.assertEqual(len(preprocessing.build_clean_dataset(raw)), 20)


class ForecastSamplesTests(unittest.TestCase):
    def test_targets_begin_after_last_observation(self):
        from src.features import make_samples
        frame = measurements()
        frame.iloc[120:135, 0] = 2
        sample = make_samples(frame, np.array([120]), lookback=120)
        np.testing.assert_allclose(sample["y"], [[0.5, 0.75, 1.25]])
        np.testing.assert_allclose(sample["sequence"][0, :, 0], 1)

    def test_missing_target_excludes_only_affected_windows(self):
        from src.features import make_samples
        frame = measurements(400)
        frame.iloc[125, 0] = np.nan
        sample = make_samples(frame, np.array([120, 300]), lookback=120)
        np.testing.assert_array_equal(sample["origins"], [300])
        np.testing.assert_allclose(sample["y"], [[0.25, 0.5, 1]])

    def test_inputs_do_not_change_when_future_measurements_change(self):
        from src.features import make_samples
        frame = measurements(11000)
        first = make_samples(frame, np.array([10800]), lookback=120)
        frame.iloc[10800:] = 999
        second = make_samples(frame, np.array([10800]), lookback=120)
        np.testing.assert_array_equal(first["sequence"], second["sequence"])
        np.testing.assert_array_equal(first["context"], second["context"])

    def test_missing_minutes_are_rejected_instead_of_compressed(self):
        from src.features import make_samples
        with self.assertRaisesRegex(ValueError, "minute"):
            make_samples(measurements().drop(measurements().index[5]), np.array([120]))

    def test_partition_origins_never_cross_target_boundary(self):
        from src.features import partition_origins
        origins = partition_origins(start=100, stop=300, lookback=120, stride=5)
        self.assertEqual(origins[0], 220)
        self.assertEqual(origins[-1], 240)


if __name__ == "__main__":
    unittest.main()
