import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ProvenanceTests(unittest.TestCase):
    def test_baseline_snapshot_survives_source_model_replacement(self):
        from src.experiments import snapshot_legacy, file_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "models", root / "run"
            source.mkdir()
            output.mkdir()
            (source / "lstm_energy_forecaster.keras").write_bytes(b"original-model")
            for filename in ("feature_scaler.joblib", "energy_target_scaler.joblib"):
                joblib.dump(StandardScaler().fit([[1., 2.], [3., 4.]]), source / filename)
            metadata = snapshot_legacy(output, source)
            (source / "lstm_energy_forecaster.keras").write_bytes(b"replacement-model")
            snapshot = output / "legacy" / "lstm_energy_forecaster.keras"
            self.assertEqual(snapshot.read_bytes(), b"original-model")
            self.assertEqual(metadata["artifacts"][snapshot.name]["sha256"], file_hash(snapshot))
            self.assertEqual(metadata["artifacts"]["feature_scaler.joblib"]["samples_seen"], 2)

    def test_pca_statistics_retain_missingness_before_complete_case_fit(self):
        from src.diagnostics import pca_report
        frame = pd.DataFrame({"a": [1., 2., None, 4.], "b": [4., 2., 1., 3.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = pca_report(frame, "check", path, complete_cases=True)
            stats = pd.read_csv(path / "feature_statistics_check.csv", index_col=0)
            self.assertEqual(stats.loc["a", "missing_fraction"], .25)
            self.assertEqual(result["samples"], 3)


if __name__ == "__main__":
    unittest.main()
