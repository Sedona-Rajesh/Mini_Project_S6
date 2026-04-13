from pathlib import Path
import json
import copy
import pandas as pd

from eeg_dss.config import load_config
from eeg_dss.training.trainer import train_model
from eeg_dss.evaluation.evaluator import evaluate_model

root = Path(r"C:/Users/SEDONA RAJESH/OneDrive/Desktop/eeg_dss")
cfg = load_config(root / "configs/config.yaml")

task = "depression"
models = ["random_forest", "hist_gradient_boosting", "stacking_ensemble"]

feat_path = Path(cfg.output_dir(task, "features")) / "features.parquet"
feature_table = pd.read_parquet(feat_path)

# Keep run reasonably fast while preserving same split/seed across models.
cfg.training["search_iterations"] = 10
cfg.training[task]["search_iterations"] = 10

run_root = root / "outputs" / "comparison" / task
results = []

for m in models:
    cfg.training["model_candidates"] = [m]
    cfg.training[task]["model_candidates"] = [m]

    art_dir = run_root / m / "artifacts"
    rep_dir = run_root / m / "reports"
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    cfg._raw["outputs"][task]["artifacts"] = str(art_dir)
    cfg._raw["outputs"][task]["reports"] = str(rep_dir)

    artifact = train_model(feature_table, task, cfg)
    summary = evaluate_model(artifact, feature_table, task, cfg)

    sm = summary.get("subject_metrics", {})
    em = summary.get("epoch_metrics", {})
    md = summary.get("training_metadata", {})
    results.append({
        "model": m,
        "subject_accuracy": sm.get("subject_accuracy"),
        "subject_f1": sm.get("subject_f1"),
        "subject_roc_auc": sm.get("subject_roc_auc"),
        "epoch_accuracy": em.get("epoch_accuracy"),
        "cv_best_score": md.get("best_cv_score"),
        "search_metric": md.get("search_metric"),
        "selected_model": md.get("selected_model"),
    })

results_sorted = sorted(results, key=lambda x: (x["subject_accuracy"] if x["subject_accuracy"] is not None else -1), reverse=True)
print("\nCONTROLLED COMPARISON (task=depression, same split seed)")
for r in results_sorted:
    print(json.dumps(r))

out_json = run_root / "comparison_summary.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(results_sorted, indent=2), encoding="utf-8")
print(f"\nSaved: {out_json}")
