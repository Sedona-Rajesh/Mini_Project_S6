# EEG Decision Support System

A modular, config-driven machine learning pipeline for binary classification of **Alzheimer's disease** and **Major Depressive Disorder** from resting-state EEG data in BIDS format.

---

## Project Structure

```
eeg_dss/
├── configs/
│   └── config.yaml               ← Single source of truth for ALL settings
├── data/
│   └── raw/
│       ├── ds004504/             ← Alzheimer BIDS dataset
│       └── ds003478/             ← Depression BIDS dataset
├── eeg_dss/
│   ├── config/
│   │   └── loader.py             ← YAML loading, Config object
│   ├── data/
│   │   ├── loader.py             ← BIDS discovery, EEG loading, channel harmonization
│   │   ├── metadata.py           ← Label inference (Alzheimer + Depression)
│   │   └── dataset_builder.py    ← Orchestrates full build pipeline
│   ├── preprocessing/
│   │   └── pipeline.py           ← Filter, bad-ch, ICA, reference, epoch
│   ├── features/
│   │   └── extractor.py          ← Statistical, spectral, complexity, connectivity
│   ├── training/
│   │   └── trainer.py            ← Random Forest + CV + threshold tuning
│   ├── evaluation/
│   │   └── evaluator.py          ← Epoch+subject metrics, plots, JSON reports
│   ├── prediction/
│   │   └── predictor.py          ← Inference on new EEG files
│   ├── visualization/
│   │   └── plots.py              ← Gauges, histograms, calibration curves
│   └── app/
│       └── streamlit_app.py      ← Clinical DSS UI
├── scripts/
│   ├── run_pipeline.py           ← Full pipeline (build + train + evaluate)
│   ├── build_features.py         ← Feature build only
│   ├── train_only.py             ← Train from cached features
│   ├── retrain_alzheimer.py      ← Force-retrain Alzheimer model
│   ├── retrain_depression.py     ← Force-retrain Depression model
│   └── benchmark.py              ← Quick sanity-check (10 balanced subjects)
├── outputs/
│   ├── alzheimer/
│   │   ├── features/             ← features.parquet, features.csv
│   │   ├── artifacts/            ← alzheimer_model.pkl, feature_names.json
│   │   └── reports/              ← confusion matrix, ROC curve, JSON metrics
│   └── depression/
│       └── ...
├── logs/
│   └── pipeline.log
├── requirements.txt
└── setup.py
```

---

## Datasets

| Task | OpenNeuro ID | Download |
|------|-------------|---------|
| Alzheimer | ds004504 | `openneuro download --dataset ds004504 --target data/raw/ds004504` |
| Depression | ds003478 | `openneuro download --dataset ds003478 --target data/raw/ds003478` |

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .

# 4. (Optional) Enable RANSAC bad-channel detection
pip install pyprep
```

---

## Terminal Commands

### Full pipeline (build → train → evaluate both tasks)
```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

### Build feature tables only (no training)
```bash
python scripts/build_features.py --config configs/config.yaml
```

### Train and evaluate from existing features
```bash
python scripts/train_only.py --config configs/config.yaml
```

### Retrain a single model from scratch
```bash
python scripts/retrain_alzheimer.py --config configs/config.yaml
python scripts/retrain_depression.py --config configs/config.yaml

# Keep cached features (only retrain the RF model)
python scripts/retrain_alzheimer.py --config configs/config.yaml --keep-features
```

### Quick benchmark (10 balanced subjects, no ICA)
```bash
python scripts/benchmark.py --config configs/config.yaml --n-subjects 10
```

### Launch the Streamlit app
```bash
streamlit run eeg_dss/app/streamlit_app.py -- --config configs/config.yaml
```

### Run only one task
```bash
python scripts/run_pipeline.py --tasks alzheimer
python scripts/run_pipeline.py --tasks depression
```

---

## Configuration

All parameters live in `configs/config.yaml`. Key sections:

```yaml
sampling:
  max_subjects_per_dataset: null   # null = use all; set e.g. 20 for quick runs
  balance_classes: true            # always sample balanced — never set false

preprocessing:
  epoch_duration_sec: 4.0
  reject_peak_to_peak_uv: 150.0    # raise if too many epochs are rejected
  run_ica: false                   # enable for production; slow

training:
  rf_param_grid:
    n_estimators: [100, 300, 500]
  tune_threshold: true
```

---

## Outputs

After a full pipeline run:

```
outputs/alzheimer/
  features/
    features.parquet          ← full epoch-level feature table
    features.csv              ← human-readable copy
  artifacts/
    alzheimer_model.pkl       ← model + scaler + threshold
    alzheimer_feature_names.json
    alzheimer_training_metadata.json
  reports/
    alzheimer_confusion_matrix.png
    alzheimer_roc_curve.png
    alzheimer_feature_importance.png
    alzheimer_subject_predictions.csv
    alzheimer_evaluation_<timestamp>.json
```

---

## Troubleshooting Guide

### 1. Missing metadata columns

**Error:**
```
ValueError: Alzheimer metadata is missing the required 'Group' column.
Found columns: ['participant_id', 'age', 'sex']
```

**Cause:** The `participants.tsv` column name differs from config.

**Fix:**
```bash
# Inspect the actual column names
head -1 data/raw/ds004504/participants.tsv
```
Then update `config.yaml`:
```yaml
alzheimer:
  group_column: "YOUR_ACTUAL_COLUMN_NAME"
```

---

**Error (Depression):**
```
Warning: BDI column 'BDI' not found. Will attempt SCID fallback.
```

**Fix:** Check what the column is actually called in `participants.tsv` and update:
```yaml
depression:
  bdi_column: "BDI_Total"   # or whatever your TSV uses
```

---

### 2. Single-class splits

**Error:**
```
RuntimeError: alzheimer: test split contains only class [0].
```

**Cause:** With very few subjects, `GroupShuffleSplit` can put all subjects of one class into the training set.

**Fix options (in order of preference):**

```yaml
# Option A: Use more subjects
sampling:
  max_subjects_per_dataset: null   # use all

# Option B: Reduce test fraction
training:
  test_size: 0.20    # was 0.25

# Option C: Ensure balanced sampling is on
sampling:
  balance_classes: true
```

---

### 3. Imbalanced subset selection

**Symptom:** Model predicts only one class; AUC ≈ 0.5.

**Root cause:** `participants.tsv` is often sorted by diagnosis group, so taking the first N subjects naively gives all positives or all controls.

**This system prevents it automatically** via `balanced_subject_sample()` when `sampling.balance_classes: true`. Verify it is enabled and check the logs for:

```
Balanced sample: 5 class-0, 5 class-1 → 10 total
```

If you still see imbalance, check that all subjects in the balanced sample successfully produced EEG epochs:
```bash
grep "failed processing" logs/pipeline.log
```

---

### 4. BIDS discovery failures

**Error:**
```
FileNotFoundError: BIDS root not found: data/raw/ds004504
```

**Fix:** Verify the dataset folder exists:
```bash
ls data/raw/
```
Update `config.yaml` to match the actual folder name:
```yaml
data:
  raw_root: "data/raw"
  alzheimer_dataset: "YOUR_FOLDER_NAME"
```

**Error:**
```
RuntimeError: No sub-* directories found under data/raw/ds004504.
```

**Fix:** The dataset may be nested. Look for where `sub-*` directories actually live:
```bash
find data/raw/ds004504 -maxdepth 3 -name "sub-*" -type d | head -5
```
Update `raw_root` to point to the parent of the `sub-*` directories.

---

### 5. EEG channel mismatch

**Error:**
```
RuntimeError: Required channels missing after harmonization: ['Cz', 'Pz']
```

**Fix:** Either remove the requirement, or check what channels are actually present:
```python
import mne
raw = mne.io.read_raw_eeglab("data/raw/ds004504/sub-001/eeg/sub-001_task-rest_eeg.set")
print(raw.ch_names)
```

Update config:
```yaml
montage:
  required_channels: []   # empty = accept whatever is available
```

**Symptom:** Very few features extracted; model performs poorly.

**Cause:** Most channels dropped during harmonization because they weren't in the standard 10-20 system.

**Fix:** Check the original channel names and see if uppercase conversion resolves it:
```python
# The loader uppercases all channels before matching to the montage
# If your channels are "EEG 001", "EEG 002" they won't match
# You may need a custom rename_map — add it to loader.py harmonize_channels()
```

---

### 6. All epochs rejected

**Warning:**
```
All epochs rejected (threshold=150.0 µV). Consider raising reject_peak_to_peak_uv.
```

**Fix:** Raise the threshold in config:
```yaml
preprocessing:
  reject_peak_to_peak_uv: 250.0   # was 150
```

Or disable rejection entirely:
```yaml
preprocessing:
  reject_peak_to_peak_uv: null
```

---

### 7. Connectivity features are slow

**Fix:** Keep connectivity disabled for development:
```yaml
features:
  connectivity: false   # enable only for production runs
```

---

### 8. Model not found when launching the app

**Error in Streamlit:**
```
No trained model found for task 'alzheimer'
```

**Fix:** Run training first:
```bash
python scripts/train_only.py --config configs/config.yaml
```

---

## Reproducibility

Every run is fully reproducible:
- `random_seed: 42` in config controls all sklearn `random_state` parameters
- Feature column order is deterministic (alphabetical within each feature group)
- Subject sampling uses a seeded `random.Random` instance
- All output artifacts are versioned with timestamps

To reproduce a specific run, use the same `config.yaml` and the same raw data. The `training_metadata.json` in `outputs/<task>/artifacts/` records the exact parameters used.
#   M i n i _ P r o j e c t _ S 6  
 