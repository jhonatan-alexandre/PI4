"""Tune forecasting hyperparameters against two chronological validation folds."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("outputs/experiments/2026-09-26"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--train-stride", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from src.tuning import run_search
    run_search(args.output, args.reference, args.epochs, args.train_stride, args.resume)


if __name__ == "__main__":
    main()
