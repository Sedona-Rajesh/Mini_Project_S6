#!/usr/bin/env python3
"""
scripts/retrain_alzheimer.py
─────────────────────────────
Force a complete rebuild of the Alzheimer model:
  * Deletes cached feature table (forces fresh preprocessing)
  * Rebuilds features from raw BIDS data
  * Trains a new model
  * Evaluates and saves artifacts

Usage
-----
    python scripts/retrain_alzheimer.py [--config configs/config.yaml]
                                        [--keep-features]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeg_dss.config import load_config
from eeg_dss.data.dataset_builder import build_dataset
from eeg_dss.evaluation.evaluator import evaluate_model
from eeg_dss.training.trainer import train_model

logger = logging.getLogger("retrain_alzheimer")
TASK = "alzheimer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain Alzheimer model from scratch")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--keep-features",
        action="store_true",
        help="Reuse cached feature table instead of rebuilding from raw EEG",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not args.keep_features:
        # Remove cached features to force full rebuild
        feat_dir = cfg.output_dir(TASK, "features")
        cache = feat_dir / "features.parquet"
        if cache.exists():
            cache.unlink()
            logger.info("Removed cached features: %s", cache)

    # Remove old model artifacts
    art_dir = cfg.output_dir(TASK, "artifacts")
    model_file = art_dir / f"{TASK}_model.pkl"
    if model_file.exists():
        model_file.unlink()
        logger.info("Removed old model: %s", model_file)

    logger.info("Retraining %s model…", TASK.upper())
    try:
        feature_table = build_dataset(TASK, cfg, force_rebuild=not args.keep_features)
        artifact = train_model(feature_table, TASK, cfg)
        summary = evaluate_model(artifact, feature_table, TASK, cfg)
        sm = summary.get("subject_metrics", {})
        print(
            f"\n✅ {TASK.upper()} retrain complete\n"
            f"   Subject AUC : {sm.get('subject_roc_auc', float('nan')):.3f}\n"
            f"   Subject F1  : {sm.get('subject_f1', float('nan')):.3f}\n"
            f"   Reports     : {cfg.output_dir(TASK, 'reports')}"
        )
    except Exception as exc:
        logger.error("Retrain FAILED: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
