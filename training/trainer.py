import numpy as np
import pandas as pd
import os
import joblib

from typing import Any
from datetime import datetime

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_model(
    feature_table: pd.DataFrame,
    task: str,
    cfg,
) -> dict[str, Any]:

    train_cfg = cfg.training
    task_train_cfg = cfg.get("training", task.lower(), default={}) or {}
    seed = cfg.seed
    out_dir = cfg.output_dir(task, "artifacts")

    # ── 1. Prepare features ─────────────────────────────────────────────
    meta_cols = {"subject_id", "label", "_label_source"}
    feature_cols = [c for c in feature_table.columns if c not in meta_cols]

    if not feature_cols:
        raise ValueError("No feature columns found")

    # Subject-level aggregation can include multiple stats for richer representation.
    agg_stats = train_cfg.get("subject_feature_aggregations", ["mean"])
    if isinstance(agg_stats, str):
        agg_stats = [agg_stats]

    subject_feats = feature_table.groupby("subject_id")[feature_cols].agg(agg_stats)
    if isinstance(subject_feats.columns, pd.MultiIndex):
        subject_feats.columns = [f"{c}_{stat}" for c, stat in subject_feats.columns]

    subject_labels = feature_table.groupby("subject_id")["label"].first()
    subject_df = subject_feats.reset_index().merge(
        subject_labels.reset_index(),
        on="subject_id",
        how="left",
    )

    # Optional training-time subject cap to stabilize small/noisy datasets.
    max_subjects = task_train_cfg.get("max_subjects", None)
    if max_subjects:
        max_subjects = int(max_subjects)
        if len(subject_df) > max_subjects:
            rng = np.random.default_rng(seed)
            if bool(task_train_cfg.get("balance_subjects", True)):
                cls0 = subject_df[subject_df["label"] == 0]["subject_id"].to_numpy()
                cls1 = subject_df[subject_df["label"] == 1]["subject_id"].to_numpy()
                per_class = max_subjects // 2
                s0 = rng.choice(cls0, size=min(per_class, len(cls0)), replace=False)
                s1 = rng.choice(cls1, size=min(per_class, len(cls1)), replace=False)
                chosen = set(np.concatenate([s0, s1]).tolist())
            else:
                chosen = set(
                    rng.choice(
                        subject_df["subject_id"].to_numpy(),
                        size=max_subjects,
                        replace=False,
                    ).tolist()
                )

            subject_df = subject_df[subject_df["subject_id"].isin(chosen)].reset_index(drop=True)

    feature_cols = [c for c in subject_df.columns if c not in {"subject_id", "label"}]

    X = subject_df[feature_cols].values.astype(np.float32)
    y = subject_df["label"].values.astype(int)
    groups = subject_df["subject_id"].values

    print(f"{task.upper()} training (SUBJECT-LEVEL): {len(X)} subjects")

    # ── 2. Train/test split ─────────────────────────────────────────────
    test_size = float(train_cfg["test_size"])
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError(f"{task}: split has only one class")

    print(f"Split: train={len(train_idx)} subjects, test={len(test_idx)} subjects")

    # ── 3. Scaling ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. TASK-SPECIFIC SETTINGS ───────────────────────────────────────
    model_candidates = task_train_cfg.get(
        "model_candidates",
        train_cfg.get("model_candidates", ["random_forest"]),
    )
    if isinstance(model_candidates, str):
        model_candidates = [model_candidates]

    if task.lower() == "depression":
        from sklearn.feature_selection import SelectKBest, f_classif

        print("Applying feature selection for depression")
        k_best = int(task_train_cfg.get("select_k_best", 100))
        cv_folds = int(task_train_cfg.get("cv_folds", train_cfg.get("cv_folds", 3)))

        selector = SelectKBest(score_func=f_classif, k=k_best)
        use_selector = True
    else:
        cv_folds = int(task_train_cfg.get("cv_folds", train_cfg.get("cv_folds", 5)))
        selector = None
        use_selector = False

    tune_threshold = bool(task_train_cfg.get("tune_threshold", train_cfg.get("tune_threshold", True)))
    threshold_metric = str(task_train_cfg.get("threshold_metric", train_cfg.get("threshold_metric", "f1"))).lower()
    threshold_grid = task_train_cfg.get(
        "threshold_grid",
        train_cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    )
    search_metric = str(task_train_cfg.get("scoring_metric", train_cfg.get("scoring_metric", "roc_auc")))
    overfit_cfg = train_cfg.get("overfit", {}) or {}
    max_cv_train_gap = float(overfit_cfg.get("max_cv_train_gap", 0.08))
    gap_penalty = float(overfit_cfg.get("gap_penalty", 0.5))

    # ── 5. Cross-validation ─────────────────────────────────────────────
    cv = StratifiedGroupKFold(n_splits=cv_folds)
    n_iter = int(task_train_cfg.get("search_iterations", train_cfg.get("search_iterations", 30)))

    candidate_specs = _build_candidate_specs(
        model_candidates=model_candidates,
        train_cfg=train_cfg,
        seed=seed,
        use_selector=use_selector,
        selector=selector,
    )
    if not candidate_specs:
        raise ValueError("No valid model candidates configured")

    iter_per_candidate = max(10, int(n_iter))
    print(
        f"Model comparison ({len(candidate_specs)} candidates, "
        f"{cv_folds} folds, scoring={search_metric})..."
    )

    best_model = None
    best_model_name = ""
    best_adjusted_score = -np.inf
    best_cv_score = -np.inf
    best_cv_gap = np.nan
    best_params: dict[str, Any] | None = None
    candidate_scores: dict[str, float] = {}
    candidate_adjusted_scores: dict[str, float] = {}

    for name, estimator, search_space in candidate_specs:
        print(f"- Searching candidate: {name}")
        search = RandomizedSearchCV(
            estimator,
            param_distributions=search_space,
            n_iter=iter_per_candidate,
            scoring=search_metric,
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            verbose=2,
            refit=False,
            return_train_score=True,
        )
        search.fit(X_train, y_train, groups=groups_train)

        idx, raw_score, adj_score, gap = _pick_cv_candidate(
            search.cv_results_,
            max_cv_train_gap=max_cv_train_gap,
            gap_penalty=gap_penalty,
        )
        params = search.cv_results_["params"][idx]
        fitted = clone(estimator).set_params(**params)
        fitted.fit(X_train, y_train)

        candidate_scores[name] = float(raw_score)
        candidate_adjusted_scores[name] = float(adj_score)

        if float(adj_score) > best_adjusted_score:
            best_adjusted_score = float(adj_score)
            best_cv_score = float(raw_score)
            best_cv_gap = float(gap)
            best_params = params
            best_model = fitted
            best_model_name = name

    if best_params is None or best_model is None:
        raise RuntimeError("Model search failed for all candidates")

    print(
        f"Best model: {best_model_name} | CV={best_cv_score:.4f} | "
        f"adjusted={best_adjusted_score:.4f} | train-cv gap={best_cv_gap:.4f}"
    )

    # ── 6. Threshold tuning ─────────────────────────────────────────────
    if tune_threshold:
        cv_proba = cross_val_predict(
            best_model,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold = _select_threshold(y_train, cv_proba, threshold_metric, threshold_grid)
        print(f"Tuned threshold ({threshold_metric}): {threshold:.3f}")
    else:
        threshold = float(task_train_cfg.get("default_threshold", train_cfg.get("default_threshold", 0.5)))

    # ── 7. Build feature names for reporting ────────────────────────────
    selected_feature_names = feature_cols
    if use_selector and hasattr(best_model, "named_steps"):
        selector = best_model.named_steps.get("selector")
        if selector is not None and hasattr(selector, "get_support"):
            mask = selector.get_support()
            selected_feature_names = [f for f, keep in zip(feature_cols, mask) if keep]

    # ── 8. Save artifacts ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "selected_feature_names": selected_feature_names,
        "threshold": threshold,
        "metadata": {
            "task": task,
            "timestamp": timestamp,
            "n_train_subjects": int(len(X_train)),
            "n_test_subjects": int(len(X_test)),
            "n_features": int(len(feature_cols)),
            "n_selected_features": int(len(selected_feature_names)),
            "best_cv_score": float(best_cv_score),
            "best_adjusted_score": float(best_adjusted_score),
            "best_cv_train_gap": float(best_cv_gap),
            "best_params": best_params,
            "cv_folds": int(cv_folds),
            "search_metric": search_metric,
            "threshold_metric": threshold_metric,
            "selected_model": best_model_name,
            "candidate_cv_scores": candidate_scores,
            "candidate_adjusted_scores": candidate_adjusted_scores,
            "max_cv_train_gap": max_cv_train_gap,
            "gap_penalty": gap_penalty,
        },
        "_test_data": {
            "X_test": X_test_scaled,
            "y_test": y_test,
            "subject_ids_test": groups[test_idx],
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifact, os.path.join(out_dir, f"{task}_model.pkl"))

    print(f"Model saved: {task}_model.pkl")

    return artifact
def load_model_artifact(path_or_task: str, cfg=None):
    import joblib
    if cfg is not None:
        model_path = cfg.output_dir(path_or_task, "artifacts") / f"{path_or_task}_model.pkl"
        return joblib.load(model_path)
    return joblib.load(path_or_task)


def _select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    threshold_grid,
) -> float:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metric_map = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    scorer = metric_map.get(metric, metric_map["f1"])

    best_t = 0.5
    best_score = -np.inf
    for t in threshold_grid:
        t = float(t)
        y_pred = (y_proba >= t).astype(int)
        score = scorer(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = t

    return float(best_t)


def _build_candidate_specs(
    model_candidates,
    train_cfg,
    seed: int,
    use_selector: bool,
    selector,
):
    specs = []

    for name in model_candidates:
        model_name = str(name).strip().lower()

        if model_name == "random_forest":
            clf = RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            params = train_cfg["rf_param_grid"]
        elif model_name in {"hist_gradient_boosting", "hgb", "boosting"}:
            clf = HistGradientBoostingClassifier(
                random_state=seed,
                early_stopping=True,
            )
            params = train_cfg.get("hgb_param_grid", {})
            if not params:
                continue
        elif model_name in {"stacking", "stacked", "stacking_ensemble"}:
            rf_base = RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
            svm_base = SVC(
                probability=True,
                kernel="rbf",
                class_weight="balanced",
                random_state=seed,
            )
            hgb_base = HistGradientBoostingClassifier(
                random_state=seed,
                early_stopping=True,
            )
            meta = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )

            clf = StackingClassifier(
                estimators=[
                    ("rf", rf_base),
                    ("svm", svm_base),
                    ("hgb", hgb_base),
                ],
                final_estimator=meta,
                stack_method="predict_proba",
                passthrough=bool(train_cfg.get("stacking_passthrough", False)),
                cv=int(train_cfg.get("stacking_cv", 3)),
                n_jobs=-1,
            )

            params = train_cfg.get("stacking_param_grid", {})
            if not params:
                params = {
                    "rf__n_estimators": [300, 500],
                    "rf__max_depth": [None, 12],
                    "svm__C": [0.5, 1.0, 2.0],
                    "svm__gamma": ["scale", 0.1],
                    "hgb__max_iter": [200, 400],
                    "hgb__learning_rate": [0.03, 0.05],
                    "final_estimator__C": [0.5, 1.0, 2.0],
                }
        else:
            continue

        if use_selector:
            estimator = Pipeline(
                steps=[
                    ("selector", selector),
                    ("clf", clf),
                ]
            )
            search_space = {f"clf__{k}": v for k, v in params.items()}
        else:
            estimator = clf
            search_space = params

        specs.append((model_name, estimator, search_space))

    return specs


def _pick_cv_candidate(cv_results, max_cv_train_gap: float, gap_penalty: float):
    test_scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    train_scores = np.asarray(cv_results.get("mean_train_score", test_scores), dtype=float)
    gaps = train_scores - test_scores
    adjusted = test_scores - gap_penalty * np.maximum(0.0, gaps - max_cv_train_gap)

    best_idx = int(np.nanargmax(adjusted))
    return best_idx, float(test_scores[best_idx]), float(adjusted[best_idx]), float(gaps[best_idx])