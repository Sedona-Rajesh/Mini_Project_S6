#!/usr/bin/env python3
"""
scripts/benchmark.py
──────────────────────
Quick end-to-end benchmark using a small balanced subject sample.

Overrides config to use at most 10 subjects per dataset (5 per class)
so the full pipeline completes in minutes, not hours.  Useful for:
  * Validating a new environment
  * Smoke-testing preprocessing changes
  * CI pipelines

Usage
-----
    python scripts/benchmark.py [--config configs/config.yaml]
                                [--n-subjects 10]
                                [--tasks alzheimer depression]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeg_dss.config import load_config, Config
from eeg_dss.data.dataset_builder import build_dataset
from eeg_dss.evaluation.evaluator import evaluate_model
from eeg_dss.training.trainer import train_model

logger = logging.getLogger("benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick EEG DSS benchmark")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument(
        "--n-subjects",
        type=int,
        default=10,
        help="Max subjects per dataset (balanced, default: 10)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["alzheimer", "depression"],
        choices=["alzheimer", "depression"],
    )
    return p.parse_args()


def _patch_config_for_benchmark(cfg: Config, n_subjects: int) -> Config:
    """Return a Config with sampling overridden for the benchmark."""
    raw = copy.deepcopy(cfg._raw)
    raw["sampling"]["max_subjects_per_dataset"] = n_subjects
    raw["sampling"]["balance_classes"] = True
    # Disable ICA and connectivity for speed
    raw["preprocessing"]["run_ica"] = False
    raw["features"]["connectivity"] = False
    patched = Config(raw, cfg.config_path)
    return patched


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bench_cfg = _patch_config_for_benchmark(cfg, args.n_subjects)

    print(
        f"\n🏃 Benchmark mode: ≤{args.n_subjects} balanced subjects per dataset\n"
        f"   ICA disabled, connectivity disabled for speed\n"
    )

    results = {}
    t_total = time.time()

    for task in args.tasks:
        t0 = time.time()
        print(f"  [{task.upper()}] building features…", end=" ", flush=True)
        try:
            # Force rebuild so benchmark always uses the limited subject set
            feature_table = build_dataset(task, bench_cfg, force_rebuild=True)
            print(f"{len(feature_table)} epochs  ", end="", flush=True)

            print("training…", end=" ", flush=True)
            artifact = train_model(feature_table, task, bench_cfg)

            print("evaluating…", end=" ", flush=True)
            summary = evaluate_model(artifact, feature_table, task, bench_cfg)

            elapsed = time.time() - t0
            sm = summary.get("subject_metrics", {})
            auc = sm.get("subject_roc_auc", float("nan"))
            f1 = sm.get("subject_f1", float("nan"))
            print(f"✅  AUC={auc:.3f}  F1={f1:.3f}  ({elapsed:.0f}s)")
            results[task] = {"auc": auc, "f1": f1, "ok": True}

        except Exception as exc:
            print(f"❌  FAILED")
            logger.error("Benchmark task %s failed: %s", task, exc, exc_info=True)
            results[task] = {"error": str(exc), "ok": False}

    total_time = time.time() - t_total
    print(f"\n  Total benchmark time: {total_time:.0f}s")

    any_failed = any(not r["ok"] for r in results.values())
    if any_failed:
        print("\n⚠️  Some tasks failed. See logs above for details.")
        sys.exit(1)
    else:
        print("\n✅  Benchmark passed.")


if __name__ == "__main__":
    main()
