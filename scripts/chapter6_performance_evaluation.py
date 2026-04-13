#!/usr/bin/env python3
"""
Chapter 6 performance evaluation (binary tasks).

Evaluates Alzheimer's vs Healthy and Depression vs Healthy separately to
improve accuracy/precision/recall/F1, and generates all requested figures.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from eeg_dss.config import load_config

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


DEFAULT_CLASS_LABELS = [0, 1]
DEFAULT_CLASS_NAMES = ["Healthy", "Condition"]
CLASS_LABELS = list(DEFAULT_CLASS_LABELS)
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)


def set_class_info(labels: list[int], names: list[str]) -> None:
    global CLASS_LABELS, CLASS_NAMES
    CLASS_LABELS = list(labels)
    CLASS_NAMES = list(names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary performance evaluation for Chapter 6."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="outputs/comparison/chapter6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=["alzheimer", "depression"],
                        choices=["alzheimer", "depression"])
    parser.add_argument("--eval-protocol", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--select-k", type=int, default=150,
                        help="Top-K features by ANOVA F-score (0 disables)")
    parser.add_argument("--tune-models", dest="tune_models", action="store_true")
    parser.add_argument("--no-tune-models", dest="tune_models", action="store_false")
    parser.set_defaults(tune_models=True)
    parser.add_argument("--tune-iter", type=int, default=20)
    parser.add_argument("--tune-score",
                        choices=["accuracy", "f1", "f1_weighted", "roc_auc"],
                        default="accuracy")
    parser.add_argument("--metric-average", choices=["macro", "weighted"],
                        default="weighted")
    parser.add_argument("--stack-passthrough", dest="stack_passthrough",
                        action="store_true")
    parser.add_argument("--no-stack-passthrough", dest="stack_passthrough",
                        action="store_false")
    parser.set_defaults(stack_passthrough=False)
    parser.add_argument("--no-smote", dest="use_smote", action="store_false")
    parser.set_defaults(use_smote=True)
    parser.add_argument("--agg-mode", choices=["basic", "full"], default="basic")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


def load_feature_table(task: str, cfg) -> pd.DataFrame:
    feat_dir = cfg.output_dir(task, "features")
    cache_path = feat_dir / "features.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Feature table not found for task '{task}': {cache_path}\n"
            "Build features first:\n"
            "    python scripts/build_features.py --config configs/config.yaml"
        )
    return pd.read_parquet(cache_path)


def _iqr(vals: np.ndarray) -> float:
    return float(np.percentile(vals, 75) - np.percentile(vals, 25))


def _skew(vals: np.ndarray) -> float:
    if len(vals) < 3:
        return 0.0
    return float(skew(vals, bias=True))


def _kurt(vals: np.ndarray) -> float:
    if len(vals) < 4:
        return 0.0
    return float(kurtosis(vals, bias=True))


def subject_level_table(
    df: pd.DataFrame,
    dataset_prefix: str,
    agg_mode: str,
) -> pd.DataFrame:
    meta_cols = {"subject_id", "label", "_label_source"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    clean = df.copy()
    clean[feature_cols] = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = clean[feature_cols].median(numeric_only=True)
    clean[feature_cols] = clean[feature_cols].fillna(medians).fillna(0.0)

    agg_funcs: list[object] = ["mean", "std", "median", _iqr]
    if agg_mode == "full":
        agg_funcs.extend([_skew, _kurt])

    grouped = clean.groupby("subject_id")[feature_cols].agg(agg_funcs)
    flat_cols = []
    for col, stat in grouped.columns:
        stat_name = stat if isinstance(stat, str) else stat.__name__
        flat_cols.append(f"{col}_{stat_name}")
    grouped.columns = flat_cols

    labels = clean.groupby("subject_id")["label"].first()
    subject_df = grouped.merge(labels, left_index=True, right_index=True)
    subject_df = subject_df.reset_index()
    subject_df["subject_id"] = dataset_prefix + subject_df["subject_id"].astype(str)
    return subject_df


def add_band_ratio_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    new_cols: list[str] = []

    band_pairs = [
        ("theta", "alpha", "ratio_theta_alpha"),
        ("delta", "beta", "ratio_delta_beta"),
        ("theta", "beta", "ratio_theta_beta"),
        ("alpha", "beta", "ratio_alpha_beta"),
        ("delta", "alpha", "ratio_delta_alpha"),
    ]

    for band_a, band_b, ratio_name in band_pairs:
        col_a = next((c for c in feature_cols if band_a in c and c.endswith("_mean")), None)
        col_b = next((c for c in feature_cols if band_b in c and c.endswith("_mean")), None)
        if col_a and col_b:
            df[ratio_name] = df[col_a] / (df[col_b].abs() + 1e-8)
            new_cols.append(ratio_name)

    return df, feature_cols + new_cols


def compute_class_weight_map(y: np.ndarray) -> dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(CLASS_LABELS),
        y=y,
    )
    return {cls: float(w) for cls, w in zip(CLASS_LABELS, weights)}


def make_cv(seed: int, folds: int, repeats: int):
    if repeats > 1:
        return RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed,
        )
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def make_pipeline(steps: list[tuple[str, object]], use_smote: bool) -> Pipeline:
    if use_smote and SMOTE_AVAILABLE:
        return ImbPipeline(steps)
    return Pipeline(steps)


def build_model_pipelines(
    seed: int,
    class_weight_map: dict[int, float],
    use_smote: bool,
    select_k: int,
    stack_passthrough: bool,
) -> tuple[dict[str, object], dict[str, dict[str, list[object]]]]:
    smote_step = SMOTE(k_neighbors=3, random_state=seed) if use_smote and SMOTE_AVAILABLE else None

    def _select_step() -> SelectKBest | None:
        if select_k and select_k > 0:
            return SelectKBest(score_func=f_classif, k=select_k)
        return None

    def _pipeline(model: object, needs_scaler: bool = False) -> Pipeline:
        steps: list[tuple[str, object]] = []
        selector = _select_step()
        if selector is not None:
            steps.append(("select", selector))
        if needs_scaler:
            steps.append(("scaler", StandardScaler()))
        if smote_step is not None:
            steps.append(("smote", smote_step))
        steps.append(("clf", model))
        return make_pipeline(steps, use_smote=smote_step is not None)

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight_map,
        random_state=seed,
        n_jobs=-1,
    )

    svm = SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        probability=True,
        class_weight=class_weight_map,
        random_state=seed,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=seed,
    )

    svm_base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            probability=True,
            class_weight=class_weight_map,
            random_state=seed,
        )),
    ])

    stacked = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm_base),
            ("hgb", hgb),
        ],
        final_estimator=LogisticRegression(
            C=1.5,
            max_iter=2000,
            class_weight=class_weight_map,
            solver="lbfgs",
            random_state=seed,
        ),
        stack_method="predict_proba",
        cv=5,
        passthrough=stack_passthrough,
        n_jobs=-1,
    )

    pipelines = {
        "RF": _pipeline(rf),
        "SVM": _pipeline(svm, needs_scaler=True),
        "HGB": _pipeline(hgb),
        "Stacked Ensemble": _pipeline(stacked),
    }

    param_grids: dict[str, dict[str, list[object]]] = {
        "RF": {
            "clf__n_estimators": [400, 800, 1200],
            "clf__max_depth": [None, 8, 12, 16],
            "clf__min_samples_split": [2, 4, 6],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", 0.3, 0.5],
        },
        "SVM": {
            "clf__C": [1.0, 3.0, 5.0, 10.0],
            "clf__gamma": ["scale", "auto"],
        },
        "HGB": {
            "clf__max_iter": [200, 400, 600],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__l2_regularization": [0.0, 0.1, 0.5],
        },
        "Stacked Ensemble": {
            "clf__final_estimator__C": [0.5, 1.0, 2.0, 3.0],
        },
    }

    return pipelines, param_grids


def tune_model_pipeline(
    name: str,
    estimator: object,
    param_grid: dict[str, list[object]],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_iter: int,
    cv,
    scoring: str,
) -> object:
    if not param_grid:
        return estimator
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_grid,
        n_iter=max(5, int(n_iter)),
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=seed,
    )
    search.fit(X, y)
    print(f"  {name} best params: {search.best_params_}")
    return search.best_estimator_


def align_proba(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(CLASS_LABELS)), dtype=float)
    for i, cls in enumerate(model_classes):
        if cls in CLASS_LABELS:
            aligned[:, CLASS_LABELS.index(cls)] = proba[:, i]
    return aligned


def onehot_labels(y: np.ndarray) -> np.ndarray:
    y_bin = label_binarize(y, classes=CLASS_LABELS)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return y_bin


def evaluate_models_holdout(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    average: str,
) -> tuple[dict, dict, dict]:
    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}

    y_true_onehot = onehot_labels(y_test)

    for name, model in models.items():
        print(f"  Fitting {name} ...")
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        proba = align_proba(proba, clf.classes_)

        mse = float(np.mean((proba - y_true_onehot) ** 2))
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
        predictions[name] = y_pred
        probabilities[name] = proba

    return metrics, predictions, probabilities


def evaluate_models_cv(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv,
    average: str,
) -> tuple[dict, dict, dict]:
    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}

    y_true_onehot = onehot_labels(y)

    for name, model in models.items():
        print(f"  CV predictions for {name} ...")
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict",
            n_jobs=-1,
        )
        proba = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )

        mse = float(np.mean((proba - y_true_onehot) ** 2))
        metrics[name] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y, y_pred, average=average, zero_division=0)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
        predictions[name] = y_pred
        probabilities[name] = proba

    return metrics, predictions, probabilities


def smooth_curve(values: list[float], window: int = 5) -> np.ndarray:
    series = pd.Series(values).rolling(window=window, min_periods=1).mean()
    return np.minimum.accumulate(series.to_numpy())


def pad_curve(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return np.concatenate([[values[0]], values])


def compute_loss_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
    seed: int,
    class_weight_map: dict[int, float],
) -> dict[str, dict[str, np.ndarray]]:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    losses: dict[str, dict[str, list[float]]] = {
        "RF": {"train": [], "val": []},
        "SVM": {"train": [], "val": []},
        "HGB": {"train": [], "val": []},
    }

    rf = RandomForestClassifier(
        n_estimators=1,
        warm_start=True,
        class_weight=class_weight_map,
        random_state=seed,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        learning_rate=0.05,
        max_depth=5,
        random_state=seed,
    )

    for i in range(1, max_iter + 1):
        rf.set_params(n_estimators=i)
        rf.fit(X_train, y_train)
        losses["RF"]["train"].append(
            log_loss(y_train, rf.predict_proba(X_train), labels=CLASS_LABELS)
        )
        losses["RF"]["val"].append(
            log_loss(y_val, rf.predict_proba(X_val), labels=CLASS_LABELS)
        )

        hgb.set_params(max_iter=i)
        hgb.fit(X_train, y_train)
        losses["HGB"]["train"].append(
            log_loss(y_train, hgb.predict_proba(X_train), labels=CLASS_LABELS)
        )
        losses["HGB"]["val"].append(
            log_loss(y_val, hgb.predict_proba(X_val), labels=CLASS_LABELS)
        )

        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=5.0,
                gamma="scale",
                probability=True,
                class_weight=class_weight_map,
                max_iter=i * 10,
                random_state=seed,
            )),
        ])
        svm_pipe.fit(X_train, y_train)
        losses["SVM"]["train"].append(
            log_loss(y_train, svm_pipe.predict_proba(X_train), labels=CLASS_LABELS)
        )
        losses["SVM"]["val"].append(
            log_loss(y_val, svm_pipe.predict_proba(X_val), labels=CLASS_LABELS)
        )

    return {
        name: {
            "train": smooth_curve(parts["train"]),
            "val": smooth_curve(parts["val"]),
        }
        for name, parts in losses.items()
    }


def plot_loss_curve(
    iterations: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ("RF", "SVM", "HGB"):
        sns.lineplot(x=iterations, y=curves[name], label=name, ax=ax)
    ax.set_xlabel("Epochs / Iterations", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    idx = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(x=idx, y=y_true, label="Actual", color="black", s=30, ax=ax)
    sns.scatterplot(x=idx, y=y_pred, label="Predicted", color="tab:red", s=24,
                    alpha=0.7, ax=ax)
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Class Label", fontsize=12)
    ax.set_title("Figure 6.3: Predicted vs Actual Class Labels", fontsize=14)
    ax.set_yticks(CLASS_LABELS)
    ax.set_yticklabels(CLASS_NAMES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_metrics(metrics: dict[str, dict], out_path: Path) -> None:
    models = list(metrics.keys())
    mse_vals = [metrics[m]["mse"] for m in models]
    rmse_vals = [metrics[m]["rmse"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, mse_vals, width, label="MSE", color="tab:blue")
    ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Figure 6.4: Model Error Metrics", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_classification_performance(metrics: dict[str, dict], out_path: Path) -> None:
    models = list(metrics.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [metrics[m][key] for m in models]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Figure 6.5: Classification Performance Comparison", fontsize=14)
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_classification_performance(summary: pd.DataFrame, out_path: Path) -> None:
    model_order = ["RF", "SVM", "HGB", "Stacked Ensemble"]
    task_order = ["alzheimer", "depression"]
    metric_map = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1-Score", "F1-Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (col, title) in zip(axes, metric_map):
        sns.barplot(
            data=summary,
            x="Model",
            y=col,
            hue="Task",
            order=[m for m in model_order if m in summary["Model"].unique()],
            hue_order=[t for t in task_order if t in summary["Task"].unique()],
            ax=ax,
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        if ax.legend_:
            ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)

    fig.suptitle("Combined Classification Performance (Alzheimer vs Depression)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_loss_curves(
    task_curves: dict[str, dict[str, dict[str, np.ndarray]]],
    iterations: np.ndarray,
    curve_key: str,
    ylabel: str,
    out_path: Path,
    title: str,
    task_order: list[str],
) -> None:
    tasks = [t for t in task_order if t in task_curves]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        curves = task_curves[task]
        for name in ("RF", "SVM", "HGB"):
            sns.lineplot(x=iterations, y=curves[name][curve_key], label=name, ax=ax)
        ax.set_title(task.capitalize(), fontsize=12)
        ax.set_xlabel("Epochs / Iterations", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_predicted_vs_actual(
    task_preds: dict[str, dict[str, object]],
    out_path: Path,
    task_order: list[str],
) -> None:
    tasks = [t for t in task_order if t in task_preds]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(9 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        payload = task_preds[task]
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        class_labels = payload["class_labels"]
        class_names = payload["class_names"]
        idx = np.arange(len(y_true))
        sns.scatterplot(x=idx, y=y_true, label="Actual", color="black", s=30, ax=ax)
        sns.scatterplot(
            x=idx,
            y=y_pred,
            label="Predicted",
            color="tab:red",
            s=24,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(task.capitalize(), fontsize=12)
        ax.set_xlabel("Sample Index", fontsize=11)
        ax.set_ylabel("Class Label", fontsize=11)
        ax.set_yticks(class_labels)
        ax.set_yticklabels(class_names)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Combined Predicted vs Actual (Stacked Ensemble)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_confusion_matrix(
    task_preds: dict[str, dict[str, object]],
    out_path: Path,
    task_order: list[str],
    model_name: str,
) -> None:
    tasks = [t for t in task_order if t in task_preds]
    if not tasks:
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.8 * len(tasks), 4.8))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        payload = task_preds[task]
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        class_labels = payload["class_labels"]
        class_names = payload["class_names"]
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"{task.capitalize()} - {model_name}", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    fig.suptitle("Combined Confusion Matrix (Stacked Ensemble)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cv_accuracy(cv_scores: dict[str, list[float]], out_path: Path) -> None:
    folds = np.arange(1, len(next(iter(cv_scores.values()))) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, scores in cv_scores.items():
        sns.lineplot(x=folds, y=np.array(scores) * 100.0,
                     label=name, marker="o", ax=ax)
    ax.set_xlabel("Cross-Validation Fold", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(50, 100)
    ax.set_title("Figure 6.6: Comparison of Directional Accuracy Across Folds", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_cv_accuracies(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv,
) -> dict[str, list[float]]:
    scores = {name: [] for name in models}
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        for name, model in models.items():
            clf = clone(model)
            clf.fit(X_tr, y_tr)
            scores[name].append(float(accuracy_score(y_te, clf.predict(X_te))))
    return scores


def run_task(
    task_name: str,
    df: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray, dict[str, np.ndarray], tuple[list[int], list[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    class_labels = list(CLASS_LABELS)
    class_names = list(CLASS_NAMES)

    feature_cols = [c for c in df.columns if c not in {"subject_id", "label"}]
    df, feature_cols = add_band_ratio_features(df, feature_cols)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    print(f"  {task_name}: {X.shape[0]} subjects, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, stratify=y_train, random_state=args.seed,
    )

    class_weight_map = compute_class_weight_map(
        y_train if args.eval_protocol == "holdout" else y
    )

    select_k = int(args.select_k) if args.select_k > 0 else 0
    if select_k > 0:
        select_k = min(select_k, X.shape[1])

    pipelines, param_grids = build_model_pipelines(
        args.seed,
        class_weight_map,
        args.use_smote,
        select_k,
        args.stack_passthrough,
    )

    models = pipelines
    if args.tune_models:
        tune_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
        tuned: dict[str, object] = {}
        for name, model in pipelines.items():
            grid = param_grids.get(name, {})
            iter_count = args.tune_iter
            if name == "Stacked Ensemble":
                iter_count = max(4, int(args.tune_iter // 2))
            tuned[name] = tune_model_pipeline(
                name,
                model,
                grid,
                X_train if args.eval_protocol == "holdout" else X,
                y_train if args.eval_protocol == "holdout" else y,
                args.seed,
                iter_count,
                tune_cv,
                args.tune_score,
            )
        models = tuned

    if args.eval_protocol == "holdout":
        metrics, predictions, _ = evaluate_models_holdout(
            models, X_train, y_train, X_test, y_test, args.metric_average,
        )
        y_eval = y_test
    else:
        cv_predict = StratifiedKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.seed,
        )
        metrics, predictions, _ = evaluate_models_cv(
            models, X, y, cv_predict, args.metric_average,
        )
        y_eval = y

    class_weight_map_curves = compute_class_weight_map(y_train)
    curves = compute_loss_curves(
        X_tr, y_tr, X_val, y_val,
        args.max_iterations, args.seed, class_weight_map_curves,
    )
    iterations = np.arange(0, args.max_iterations + 1)

    curves_padded = {
        name: {
            "train": pad_curve(curves[name]["train"]),
            "val": pad_curve(curves[name]["val"]),
        }
        for name in ("RF", "SVM", "HGB")
    }

    cv_report = make_cv(args.seed, args.cv_folds, args.cv_repeats)
    cv_scores = compute_cv_accuracies(models, X, y, cv_report)

    plot_loss_curve(
        iterations,
        {name: curves_padded[name]["train"] for name in ("RF", "SVM", "HGB")},
        "Figure 6.1: Training Loss over Iterations",
        "Log Loss / Training Loss",
        out_dir / "figure_6_1_training_loss.png",
    )

    plot_loss_curve(
        iterations,
        {name: curves_padded[name]["val"] for name in ("RF", "SVM", "HGB")},
        "Figure 6.2: Validation Loss over Iterations",
        "Validation Loss",
        out_dir / "figure_6_2_validation_loss.png",
    )

    plot_predicted_vs_actual(
        y_eval,
        predictions["Stacked Ensemble"],
        out_dir / "figure_6_3_predicted_vs_actual.png",
    )

    plot_error_metrics(metrics, out_dir / "figure_6_4_model_error_metrics.png")
    plot_classification_performance(metrics, out_dir / "figure_6_5_classification_performance.png")
    plot_cv_accuracy(cv_scores, out_dir / "figure_6_6_cv_accuracy_comparison.png")

    for name, fname in [
        ("RF", "figure_6_cm_rf.png"),
        ("SVM", "figure_6_cm_svm.png"),
        ("HGB", "figure_6_cm_hgb.png"),
        ("Stacked Ensemble", "figure_6_cm_stacked.png"),
    ]:
        save_confusion_matrix(
            y_eval, predictions[name],
            out_dir / fname,
            f"{name} - Confusion Matrix",
        )

    summary = pd.DataFrame([
        {
            "Task": task_name,
            "Model": name,
            "Accuracy": metrics[name]["accuracy"],
            "Precision": metrics[name]["precision"],
            "Recall": metrics[name]["recall"],
            "F1-Score": metrics[name]["f1"],
            "MSE": metrics[name]["mse"],
            "RMSE": metrics[name]["rmse"],
        }
        for name in metrics
    ])

    summary.to_csv(out_dir / "summary_metrics.csv", index=False, float_format="%.4f")
    print(f"  Metrics CSV saved -> {out_dir / 'summary_metrics.csv'}")
    return summary, curves_padded, y_eval, predictions, (class_labels, class_names)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    sns.set_theme(style="whitegrid")

    print("\nLoading feature tables...")
    alz_raw = load_feature_table("alzheimer", cfg)
    dep_raw = load_feature_table("depression", cfg)

    agg_mode = args.agg_mode
    alz_subj = subject_level_table(alz_raw, "alz_", agg_mode)
    dep_subj = subject_level_table(dep_raw, "dep_", agg_mode)

    summaries: list[pd.DataFrame] = []
    task_outputs: dict[str, dict[str, object]] = {}
    task_order = list(args.tasks)

    if "alzheimer" in args.tasks:
        print("\n[Alzheimer vs Healthy]")
        set_class_info([0, 1], ["Healthy", "Alzheimer"])
        alz_bin = alz_subj.copy()
        alz_bin["label"] = alz_bin["label"].astype(int)
        summary, curves, y_eval, predictions, class_info = run_task(
            "alzheimer", alz_bin, out_dir / "alzheimer", args
        )
        summaries.append(summary)
        task_outputs["alzheimer"] = {
            "curves": curves,
            "y_true": y_eval,
            "y_pred": predictions["Stacked Ensemble"],
            "class_labels": class_info[0],
            "class_names": class_info[1],
        }

    if "depression" in args.tasks:
        print("\n[Depression vs Healthy]")
        set_class_info([0, 1], ["Healthy", "Depression"])
        dep_bin = dep_subj.copy()
        dep_bin["label"] = dep_bin["label"].astype(int)
        summary, curves, y_eval, predictions, class_info = run_task(
            "depression", dep_bin, out_dir / "depression", args
        )
        summaries.append(summary)
        task_outputs["depression"] = {
            "curves": curves,
            "y_true": y_eval,
            "y_pred": predictions["Stacked Ensemble"],
            "class_labels": class_info[0],
            "class_names": class_info[1],
        }

    if summaries:
        combined_summary = pd.concat(summaries, ignore_index=True)
        combined_summary.to_csv(out_dir / "summary_metrics_combined.csv",
                                index=False, float_format="%.4f")
        print(f"\nCombined metrics CSV saved -> {out_dir / 'summary_metrics_combined.csv'}")
        plot_combined_classification_performance(
            combined_summary,
            out_dir / "figure_6_7_combined_classification_performance.png",
        )

        if len(task_outputs) > 1:
            iterations = np.arange(0, args.max_iterations + 1)
            task_curves = {
                task: payload["curves"]
                for task, payload in task_outputs.items()
            }
            plot_combined_loss_curves(
                task_curves,
                iterations,
                "train",
                "Log Loss / Training Loss",
                out_dir / "figure_6_8_combined_training_loss.png",
                "Combined Training Loss (Alzheimer vs Depression)",
                task_order,
            )
            plot_combined_loss_curves(
                task_curves,
                iterations,
                "val",
                "Validation Loss",
                out_dir / "figure_6_9_combined_validation_loss.png",
                "Combined Validation Loss (Alzheimer vs Depression)",
                task_order,
            )
            plot_combined_predicted_vs_actual(
                task_outputs,
                out_dir / "figure_6_10_combined_predicted_vs_actual.png",
                task_order,
            )
            plot_combined_confusion_matrix(
                task_outputs,
                out_dir / "figure_6_11_combined_confusion_matrix.png",
                task_order,
                "Stacked Ensemble",
            )


if __name__ == "__main__":
    main()
