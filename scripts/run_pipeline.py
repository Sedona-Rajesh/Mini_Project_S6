#!/usr/bin/env python3
"""
scripts/run_pipeline.py
───────────────────────
Run the complete EEG DSS pipeline end-to-end:
  1. Build feature tables for Alzheimer and Depression datasets
  2. Train both models
  3. Evaluate both models
  4. Print a final summary

Usage
-----
    python scripts/run_pipeline.py [--config configs/config.yaml] [--force-rebuild]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeg_dss.config import load_config
from eeg_dss.data.dataset_builder import build_dataset
from eeg_dss.evaluation.evaluator import evaluate_model
from eeg_dss.training.trainer import train_model

logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full EEG DSS pipeline")
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore cached feature tables and rebuild from raw data",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["alzheimer", "depression"],
        choices=["alzheimer", "depression"],
        help="Which tasks to run (default: both)",
    )
    return p.parse_args()


def run_task(task: str, cfg, force_rebuild: bool) -> dict:
    """Run build → train → evaluate for a single task. Return eval summary."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("TASK: %s", task.upper())
    logger.info("=" * 60)

    # ── Build features ────────────────────────────────────────────────────
    logger.info("[1/3] Building feature table…")
    feature_table = build_dataset(task, cfg, force_rebuild=force_rebuild)

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("[2/3] Training model…")
    artifact = train_model(feature_table, task, cfg)

    # ── Evaluate ──────────────────────────────────────────────────────────
    logger.info("[3/3] Evaluating model…")
    summary = evaluate_model(artifact, feature_table, task, cfg)

    elapsed = time.time() - t0
    logger.info(
        "%s completed in %.1f s  |  subject AUC = %.3f",
        task.upper(),
        elapsed,
        summary.get("subject_metrics", {}).get("subject_roc_auc", float("nan")),
    )
    return summary


def _check_minimum_targets(all_summaries: dict[str, dict], cfg) -> list[str]:
    """Return list of target violations using cfg.minimum_targets if present."""
    targets = cfg.get("minimum_targets", default={}) or {}
    violations: list[str] = []

    for task, task_targets in targets.items():
        summary = all_summaries.get(task)
        if not summary or "error" in summary:
            continue

        subject_metrics = summary.get("subject_metrics", {})
        for metric_name, min_value in (task_targets or {}).items():
            actual = subject_metrics.get(metric_name)
            if actual is None:
                violations.append(
                    f"{task}.{metric_name}: missing (target >= {min_value})"
                )
                continue
            if float(actual) + 1e-12 < float(min_value):
                violations.append(
                    f"{task}.{metric_name}: {actual:.3f} < target {float(min_value):.3f}"
                )

    return violations


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    all_summaries: dict[str, dict] = {}

    for task in args.tasks:
        try:
            summary = run_task(task, cfg, force_rebuild=args.force_rebuild)
            all_summaries[task] = summary
        except Exception as exc:
            logger.error("Task %s FAILED: %s", task, exc, exc_info=True)
            all_summaries[task] = {"error": str(exc)}

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for task, summary in all_summaries.items():
        if "error" in summary:
            print(f"  {task.upper():<14}  FAILED — {summary['error'][:60]}")
        else:
            sm = summary.get("subject_metrics", {})
            em = summary.get("epoch_metrics", {})
            print(
                f"  {task.upper():<14}  "
                f"subj AUC={sm.get('subject_roc_auc', float('nan')):.3f}  "
                f"subj F1={sm.get('subject_f1', float('nan')):.3f}  "
                f"epoch AUC={em.get('epoch_roc_auc', float('nan')):.3f}"
            )
    print("=" * 60)

    violations = _check_minimum_targets(all_summaries, cfg)
    if violations:
        print("\nMinimum target check FAILED:")
        for v in violations:
            print(f"  - {v}")
    else:
        if cfg.get("minimum_targets", default=None):
            print("Minimum target check PASSED")

    # Exit non-zero if any task failed
    if any("error" in s for s in all_summaries.values()) or violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
