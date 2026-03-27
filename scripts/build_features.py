#!/usr/bin/env python3
"""
scripts/build_features.py
──────────────────────────
Build (or rebuild) feature tables from raw BIDS EEG data.
Does NOT train any model.

Usage
-----
    python scripts/build_features.py [--config configs/config.yaml]
                                     [--tasks alzheimer depression]
                                     [--force-rebuild]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeg_dss.config import load_config
from eeg_dss.data.dataset_builder import build_dataset

logger = logging.getLogger("build_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EEG feature tables")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["alzheimer", "depression"],
        choices=["alzheimer", "depression"],
    )
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    for task in args.tasks:
        logger.info("Building features for task: %s", task)
        try:
            df = build_dataset(task, cfg, force_rebuild=args.force_rebuild)
            out_dir = cfg.output_dir(task, "features")
            logger.info(
                "%s: %d epochs, %d features → %s",
                task.upper(),
                len(df),
                df.shape[1] - 2,
                out_dir / "features.parquet",
            )
        except Exception as exc:
            logger.error("Feature build for %s FAILED: %s", task, exc, exc_info=True)
            sys.exit(1)

    print("Feature build complete.")


if __name__ == "__main__":
    main()
