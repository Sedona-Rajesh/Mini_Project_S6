#!/usr/bin/env python3
"""
scripts/train_only.py
──────────────────────
Train and evaluate models from already-built feature tables.
Fails with a clear message if features have not been built yet.

Usage
-----
    python scripts/train_only.py [--config configs/config.yaml]
                                 [--tasks alzheimer depression]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeg_dss.config import load_config
from eeg_dss.evaluation.evaluator import evaluate_model
from eeg_dss.training.trainer import train_model

import pandas as pd

logger = logging.getLogger("train_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EEG models from existing features")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["alzheimer", "depression"],
        choices=["alzheimer", "depression"],
    )
    return p.parse_args()


def _load_features(task: str, cfg) -> pd.DataFrame:
    """Load cached feature parquet; raise with helpful message if absent."""
    feat_dir = cfg.output_dir(task, "features")
    cache_path = feat_dir / "features.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature table not found for task '{task}': {cache_path}\n"
            "Build features first:\n"
            "    python scripts/build_features.py --config configs/config.yaml"
        )
    df = pd.read_parquet(cache_path)
    logger.info("Loaded %s features: %d epochs × %d cols", task, *df.shape)
    return df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    failed = []
    for task in args.tasks:
        try:
            feature_table = _load_features(task, cfg)
            artifact = train_model(feature_table, task, cfg)
            summary = evaluate_model(artifact, feature_table, task, cfg)
            sm = summary.get("subject_metrics", {})
            print(
                f"  {task.upper():<14}  "
                f"subj AUC={sm.get('subject_roc_auc', float('nan')):.3f}  "
                f"subj F1={sm.get('subject_f1', float('nan')):.3f}"
            )
        except Exception as exc:
            logger.error("Task %s FAILED: %s", task, exc, exc_info=True)
            failed.append(task)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
