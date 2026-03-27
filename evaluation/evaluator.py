"""
eeg_dss/evaluation/evaluator.py
─────────────────────────────────
Compute and persist evaluation metrics at both epoch-level and
subject-level (majority-vote or mean-proba aggregation).

Outputs
-------
  * JSON: metrics summary
  * PNG:  confusion matrix, ROC curve, feature importance bar chart
  * CSV:  per-subject predictions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_dss.config import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    artifact: dict[str, Any],
    feature_table: pd.DataFrame,
    task: str,
    cfg: Config,
) -> dict[str, Any]:
    """
    Evaluate the trained model on the held-out test set.

    Parameters
    ----------
    artifact:
        Output of ``train_model``.
    feature_table:
        Full feature table (train+test); test indices are in artifact.
    task:
        ``"alzheimer"`` or ``"depression"``.
    cfg:
        Loaded Config.

    Returns
    -------
    dict of evaluation results.
    """
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact.get("selected_feature_names", artifact["feature_names"])
    threshold = artifact["threshold"]
    test_data = artifact["_test_data"]

    X_test = test_data["X_test"]     # already scaled
    y_test = test_data["y_test"]
    subject_ids = test_data["subject_ids_test"]

    out_dir = cfg.output_dir(task, "reports")
    eval_cfg = cfg.evaluation
    agg_method = eval_cfg.get("subject_aggregation", "majority_vote")

    # ── Epoch-level predictions ───────────────────────────────────────────
    proba = model.predict_proba(X_test)[:, 1]
    epoch_preds = (proba >= threshold).astype(int)

    epoch_metrics = _compute_metrics(y_test, epoch_preds, proba, "epoch")
    logger.info("Epoch-level metrics: %s", _fmt_metrics(epoch_metrics))

    # ── Subject-level predictions ─────────────────────────────────────────
    subj_df = _aggregate_subject_predictions(
        subject_ids, y_test, epoch_preds, proba, agg_method, threshold
    )
    subj_metrics = _compute_metrics(
        subj_df["true_label"].values,
        subj_df["predicted_label"].values,
        subj_df["mean_proba"].values,
        "subject",
    )
    logger.info("Subject-level metrics: %s", _fmt_metrics(subj_metrics))

    # ── Save plots ────────────────────────────────────────────────────────
    if eval_cfg.get("save_confusion_matrix", True):
        _save_confusion_matrix(
            subj_df["true_label"].values,
            subj_df["predicted_label"].values,
            task,
            out_dir,
        )

    if eval_cfg.get("save_roc_curve", True):
        _save_roc_curve(
            subj_df["true_label"].values,
            subj_df["mean_proba"].values,
            task,
            out_dir,
        )

    if eval_cfg.get("save_feature_importance", True):
        _save_feature_importance(
            model,
            feature_names,
            task,
            out_dir,
            top_n=int(eval_cfg.get("top_n_features", 20)),
        )

    # ── Save per-subject prediction CSV ──────────────────────────────────
    subj_csv = out_dir / f"{task}_subject_predictions.csv"
    subj_df.to_csv(subj_csv, index=False)

    # ── Save JSON summary ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "task": task,
        "timestamp": timestamp,
        "threshold": float(threshold),
        "aggregation": agg_method,
        "epoch_metrics": epoch_metrics,
        "subject_metrics": subj_metrics,
        "training_metadata": artifact["metadata"],
    }

    json_path = out_dir / f"{task}_evaluation_{timestamp}.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Evaluation report saved: %s", json_path)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    level: str,
) -> dict[str, float]:
    """Return classification metrics dict."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            "%s-level evaluation: only class %s in y_true. "
            "ROC-AUC and some metrics will be unavailable.",
            level,
            unique_classes,
        )
        metrics[f"{level}_accuracy"] = float(accuracy_score(y_true, y_pred))
        return metrics

    metrics[f"{level}_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{level}_precision"] = float(
        precision_score(y_true, y_pred, zero_division=0)
    )
    metrics[f"{level}_recall"] = float(
        recall_score(y_true, y_pred, zero_division=0)
    )
    metrics[f"{level}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        metrics[f"{level}_roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics[f"{level}_roc_auc"] = float("nan")

    return metrics


def _aggregate_subject_predictions(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    epoch_preds: np.ndarray,
    probas: np.ndarray,
    method: str,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate epoch-level predictions to subject-level."""
    df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "true_label": y_true,
            "epoch_pred": epoch_preds,
            "proba": probas,
        }
    )

    rows = []
    for subj, grp in df.groupby("subject_id"):
        true_label = grp["true_label"].iloc[0]
        mean_proba = float(grp["proba"].mean())

        if method == "majority_vote":
            predicted_label = int(grp["epoch_pred"].mode().iloc[0])
        else:  # mean_proba
            predicted_label = int(mean_proba >= threshold)

        rows.append(
            {
                "subject_id": subj,
                "true_label": int(true_label),
                "predicted_label": predicted_label,
                "mean_proba": mean_proba,
                "n_epochs": len(grp),
            }
        )

    return pd.DataFrame(rows)


def _save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, task: str, out_dir: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{task.capitalize()} — Subject-level Confusion Matrix")
    fig.tight_layout()
    path = out_dir / f"{task}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", path)


def _save_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, task: str, out_dir: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, roc_auc_score

    if len(np.unique(y_true)) < 2:
        logger.warning("ROC curve skipped: only one class in test set.")
        return

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"{task.capitalize()} — ROC Curve (AUC={auc:.3f})")
    fig.tight_layout()
    path = out_dir / f"{task}_roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved: %s", path)


def _save_feature_importance(
    model,
    feature_names: list[str],
    task: str,
    out_dir: Path,
    top_n: int = 20,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = model
    if hasattr(model, "named_steps"):
        if "rf" in model.named_steps:
            estimator = model.named_steps["rf"]
        elif "clf" in model.named_steps:
            estimator = model.named_steps["clf"]

    if not hasattr(estimator, "feature_importances_"):
        logger.warning("Model does not expose feature importances; skipping chart.")
        return

    importances = estimator.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(len(top_names)), top_vals[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"{task.capitalize()} — Top {top_n} Features")
    fig.tight_layout()
    path = out_dir / f"{task}_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance chart saved: %s", path)


def _fmt_metrics(m: dict) -> str:
    return ", ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in m.items())
