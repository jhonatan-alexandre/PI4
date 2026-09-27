"""Audit data, fit PCA, compare historical coverage and retrain energy models.

Example: python experiments.py --output outputs/experiments/run_01 --epochs 25
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--train-stride", type=int, default=15)
    parser.add_argument("--neural-days", type=int, default=365)
    parser.add_argument("--reference-rows", type=int, default=200000)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from src.experiments import run
    run(output=args.output, epochs=args.epochs, train_stride=args.train_stride,
        neural_days=args.neural_days, reference_rows=args.reference_rows,
        lookback=args.lookback, seed=args.seed)


if __name__ == "__main__":
    main()
