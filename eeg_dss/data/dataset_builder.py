"""
eeg_dss/data/dataset_builder.py
────────────────────────────────
Orchestrates the full data → features pipeline for a single dataset.

  1. Discover subjects from BIDS root
  2. Load + validate participants metadata
  3. Infer binary labels
  4. (Optional) Balanced subject sampling
  5. Per-subject: load EEG → harmonize → preprocess → epoch → features
  6. Concatenate and save feature table
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from eeg_dss.config import Config
from eeg_dss.data.loader import (
    build_run_index,
    discover_bids_subjects,
    harmonize_channels,
    load_raw,
)
from eeg_dss.data.metadata import (
    balanced_subject_sample,
    infer_alzheimer_labels,
    infer_depression_labels,
    load_participants_tsv,
)
from eeg_dss.features.extractor import extract_features
from eeg_dss.preprocessing.pipeline import make_epochs, preprocess_raw

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(
    task: str,          # "alzheimer" | "depression"
    cfg: Config,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build (or load cached) feature table for the specified task.

    Parameters
    ----------
    task:
        Either ``"alzheimer"`` or ``"depression"``.
    cfg:
        Loaded Config.
    force_rebuild:
        If True, ignore cached CSVs and reprocess from raw data.

    Returns
    -------
    pd.DataFrame with columns: subject_id, label, <feature_columns>
    """
    if task not in ("alzheimer", "depression"):
        raise ValueError(f"Unknown task '{task}'. Must be 'alzheimer' or 'depression'.")

    dataset_key = f"{task}_dataset"
    bids_root = cfg.data_root(dataset_key)
    out_dir = cfg.output_dir(task, "features")
    cache_path = out_dir / "features.parquet"

    # ── Load cache if available ───────────────────────────────────────────
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached features from %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Cached feature table: %d epochs, %d features", *df.shape)
        return df

    logger.info("Building %s dataset from %s", task.upper(), bids_root)

    # ── 1. Discover subjects ──────────────────────────────────────────────
    all_subjects = discover_bids_subjects(bids_root)

    # ── 2. Load metadata and infer labels ─────────────────────────────────
    participants = load_participants_tsv(bids_root)

    if task == "alzheimer":
        label_col = cfg.alzheimer["label_column_name"]
        metadata = infer_alzheimer_labels(participants, cfg)
    else:
        label_col = cfg.depression["label_column_name"]
        metadata = infer_depression_labels(participants, cfg)

    # ── 3. Intersect subjects with labeled metadata ───────────────────────
    labeled_subjects = [s for s in all_subjects if s in metadata.index]
    if not labeled_subjects:
        raise RuntimeError(
            f"No subjects from BIDS root have valid labels for task '{task}'.\n"
            f"BIDS subjects found: {len(all_subjects)}\n"
            f"Metadata subjects after labeling: {len(metadata)}\n"
            "Check participants.tsv and the labeling config section."
        )

    logger.info(
        "%s: %d subjects have both EEG data and valid labels",
        task.upper(),
        len(labeled_subjects),
    )

    # ── 4. Optional balanced sampling ─────────────────────────────────────
    max_subs = cfg.sampling.get("max_subjects_per_dataset", None)
    task_caps = cfg.sampling.get("task_max_subjects", {}) or {}
    if task in task_caps:
        max_subs = task_caps[task]
    if max_subs and len(labeled_subjects) > int(max_subs):
        if cfg.sampling.get("balance_classes", True):
            labeled_subjects = balanced_subject_sample(
                subjects=labeled_subjects,
                metadata=metadata,
                label_col=label_col,
                max_subjects=int(max_subs),
                seed=cfg.seed,
            )
        else:
            labeled_subjects = labeled_subjects[: int(max_subs)]
            logger.warning(
                "Limiting to first %d subjects without balancing. "
                "Set sampling.balance_classes=true to avoid class bias.",
                max_subs,
            )

    # ── 5. Build run index ────────────────────────────────────────────────
    run_index = build_run_index(bids_root, labeled_subjects)

    # ── 6. Per-subject feature extraction ─────────────────────────────────
    montage_name = cfg.montage.get("target", "standard_1020")
    required_chs = cfg.montage.get("required_channels", []) or []

    all_features: list[pd.DataFrame] = []
    failed_subjects: list[str] = []
    subject_epoch_counts: dict[str, int] = {}

    for subject_id, group in run_index.groupby("subject_id"):
        subject_label = int(metadata.loc[subject_id, label_col])
        subject_feats = _process_subject(
            subject_id=subject_id,
            file_paths=group["file_path"].tolist(),
            label=subject_label,
            cfg=cfg,
            montage_name=montage_name,
            required_chs=required_chs,
        )
        if subject_feats is None:
            failed_subjects.append(subject_id)
        else:
            subject_epoch_counts[subject_id] = int(len(subject_feats))
            all_features.append(subject_feats)

    if failed_subjects:
        logger.warning(
            "%d subject(s) failed processing and were excluded: %s",
            len(failed_subjects),
            failed_subjects[:10],
        )

    if not all_features:
        raise RuntimeError(
            f"No valid features produced for task '{task}'.\n"
            f"All {len(labeled_subjects)} subjects failed processing.\n"
            "Check preprocessing settings and EEG file integrity."
        )

    # ── 7. Concatenate and validate ───────────────────────────────────────
    feature_table = pd.concat(all_features, ignore_index=True)
    feature_table = feature_table.reset_index(drop=True)

    n_epochs = len(feature_table)
    n_subjects = feature_table["subject_id"].nunique()
    label_counts = feature_table.groupby("subject_id")["label"].first().value_counts()
    epoch_counts = feature_table["label"].value_counts().to_dict()

    logger.info(
        "%s feature table: %d epochs from %d subjects | labels: %s",
        task.upper(),
        n_epochs,
        n_subjects,
        dict(label_counts),
    )
    logger.info(
        "%s epoch distribution by class: %s",
        task.upper(),
        epoch_counts,
    )
    if subject_epoch_counts:
        min_ep = min(subject_epoch_counts.values())
        med_ep = int(pd.Series(subject_epoch_counts).median())
        max_ep = max(subject_epoch_counts.values())
        logger.info(
            "%s subject epoch stats: min=%d median=%d max=%d",
            task.upper(),
            min_ep,
            med_ep,
            max_ep,
        )

    _check_class_viability(label_counts, task)

    # ── 8. Save to disk ───────────────────────────────────────────────────
    feature_table.to_parquet(cache_path, index=False)
    logger.info("Feature table saved to %s", cache_path)

    # Also save a human-readable CSV for inspection
    csv_path = out_dir / "features.csv"
    feature_table.to_csv(csv_path, index=False)

    return feature_table


# ──────────────────────────────────────────────────────────────────────────────
def _process_subject(
    subject_id: str,
    file_paths: list[str],
    label: int,
    cfg: Config,
    montage_name: str,
    required_chs: list[str],
) -> pd.DataFrame | None:
    """
    Load, preprocess, epoch, and extract features for one subject.

    Returns a DataFrame or None on failure.
    """
    import mne

    subject_epoch_blocks: list[pd.DataFrame] = []
    min_epochs_per_run = int(cfg.preprocessing.get("min_epochs_per_run", 3))
    min_epochs_per_subject = int(cfg.preprocessing.get("min_epochs_per_subject", 10))

    for file_path in file_paths:
        fp = Path(file_path)
        try:
            raw = load_raw(fp)
        except Exception as exc:
            logger.warning("Subject %s: failed to load %s — %s", subject_id, fp.name, exc)
            continue

        # Harmonize channels
        try:
            raw = harmonize_channels(
                raw,
                target_montage_name=montage_name,
                required_channels=required_chs if required_chs else None,
            )
        except RuntimeError as exc:
            logger.warning(
                "Subject %s: channel harmonization failed — %s", subject_id, exc
            )
            continue

        if len(raw.ch_names) == 0:
            logger.warning(
                "Subject %s: no channels remain after harmonization — skipping run",
                subject_id,
            )
            continue

        # Preprocess
        try:
            raw_clean = preprocess_raw(raw, cfg)
        except Exception as exc:
            logger.warning(
                "Subject %s: preprocessing failed for %s — %s",
                subject_id, fp.name, exc,
            )
            continue

        # Epoch
        epochs = make_epochs(raw_clean, cfg)
        if epochs is None or len(epochs) == 0:
            logger.warning(
                "Subject %s: no valid epochs from %s", subject_id, fp.name
            )
            continue
        if len(epochs) < min_epochs_per_run:
            logger.warning(
                "Subject %s: only %d epochs from %s (< min_epochs_per_run=%d) — skipping run",
                subject_id,
                len(epochs),
                fp.name,
                min_epochs_per_run,
            )
            continue

        # Extract features
        try:
            feats = extract_features(epochs, cfg, subject_id=subject_id)
            feats["label"] = label
            subject_epoch_blocks.append(feats)
        except Exception as exc:
            logger.warning(
                "Subject %s: feature extraction failed for %s — %s",
                subject_id, fp.name, exc,
            )
            continue

    if not subject_epoch_blocks:
        return None
    subject_df = pd.concat(subject_epoch_blocks, ignore_index=True)
    if len(subject_df) < min_epochs_per_subject:
        logger.warning(
            "Subject %s excluded: only %d epochs across runs (< min_epochs_per_subject=%d)",
            subject_id,
            len(subject_df),
            min_epochs_per_subject,
        )
        return None

    return subject_df


# ──────────────────────────────────────────────────────────────────────────────
def _check_class_viability(
    label_counts: pd.Series, task: str
) -> None:
    """
    Raise a descriptive error if only one class is present after processing.
    """
    if len(label_counts) < 2:
        present_classes = list(label_counts.index)
        raise RuntimeError(
            f"Only one class is present in the {task} feature table: {present_classes}\n"
            "Possible causes:\n"
            "  1. The dataset has very few subjects of one class.\n"
            "  2. balanced_subject_sample failed silently.\n"
            "  3. All subjects of one class failed EEG processing.\n"
            "Fix: check failed_subjects warnings above, expand the dataset, "
            "or relax preprocessing thresholds."
        )
