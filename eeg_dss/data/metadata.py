"""
eeg_dss/data/metadata.py
────────────────────────
Parse participants.tsv from a BIDS dataset and infer binary labels.

Two labeling strategies are implemented:
  * Alzheimer  — Group column: A → 1, C → 0, F → exclude
  * Depression — BDI threshold primary, SCID fallback
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from eeg_dss.config import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def load_participants_tsv(bids_root: Path) -> pd.DataFrame:
    """
    Load participants.tsv from a BIDS root.

    Returns
    -------
    pd.DataFrame with participant_id as index.
    """
    tsv_path = bids_root / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"participants.tsv not found: {tsv_path}\n"
            "Every BIDS dataset must have a participants.tsv at its root."
        )

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    if "participant_id" not in df.columns:
        raise ValueError(
            f"participants.tsv is missing required 'participant_id' column.\n"
            f"Found columns: {list(df.columns)}\nFile: {tsv_path}"
        )

    # Normalise participant_id → strip 'sub-' prefix for matching
    df["participant_id"] = df["participant_id"].str.strip()
    df["subject_id"] = df["participant_id"].str.replace(
        r"^sub-", "", regex=True
    )
    df = df.set_index("subject_id", drop=False)

    logger.info(
        "Loaded participants.tsv: %d rows, columns=%s",
        len(df),
        list(df.columns),
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
def infer_alzheimer_labels(
    participants: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Add binary label column to *participants* for the Alzheimer task.

    Label rules (from config):
        positive_class_values → 1
        control_class_values  → 0
        exclude_values        → excluded (row dropped)
        unknown               → excluded with warning

    Parameters
    ----------
    participants:
        DataFrame from ``load_participants_tsv``.
    cfg:
        Loaded Config.

    Returns
    -------
    pd.DataFrame with only labeled rows and a new ``label`` column (int).
    """
    alz_cfg = cfg.alzheimer
    group_col = alz_cfg["group_column"]
    positive = [str(v).strip() for v in alz_cfg["positive_class_values"]]
    control = [str(v).strip() for v in alz_cfg["control_class_values"]]
    exclude = [str(v).strip() for v in alz_cfg.get("exclude_values", [])]
    label_col = alz_cfg["label_column_name"]

    if group_col not in participants.columns:
        raise ValueError(
            f"Alzheimer metadata is missing the required '{group_col}' column.\n"
            f"Found columns: {list(participants.columns)}\n"
            "Check config.yaml → alzheimer.group_column and your participants.tsv."
        )

    df = participants.copy()
    df[group_col] = df[group_col].astype(str).str.strip()

    labels: list[Optional[int]] = []
    excluded_count = 0
    unknown_values: list[str] = []

    for _, row in df.iterrows():
        val = row[group_col]
        if val in positive:
            labels.append(1)
        elif val in control:
            labels.append(0)
        elif val in exclude:
            labels.append(None)
            excluded_count += 1
        else:
            labels.append(None)
            excluded_count += 1
            unknown_values.append(val)

    df[label_col] = labels

    if unknown_values:
        unique_unknown = sorted(set(unknown_values))
        logger.warning(
            "Alzheimer labels: %d row(s) excluded due to unknown Group values: %s\n"
            "Expected positives=%s, controls=%s, excludes=%s",
            len(unknown_values),
            unique_unknown,
            positive,
            control,
            exclude,
        )

    if excluded_count > 0:
        logger.info(
            "Alzheimer labels: %d row(s) excluded (exclude_values or unknown)",
            excluded_count,
        )

    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)

    _log_class_distribution(df, label_col, "Alzheimer")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def infer_depression_labels(
    participants: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Add binary label column to *participants* for the Depression task.

    Labeling strategy (config-driven):
        Primary  : BDI >= threshold → 1; BDI < threshold → 0
        Fallback : SCID diagnosis fields when BDI is missing
        Exclude  : rows where neither BDI nor SCID is usable

    Parameters
    ----------
    participants:
        DataFrame from ``load_participants_tsv``.
    cfg:
        Loaded Config.

    Returns
    -------
    pd.DataFrame with only labeled rows and a new ``label`` column (int).
    """
    dep_cfg = cfg.depression
    bdi_col = dep_cfg["bdi_column"]
    bdi_thresh = float(dep_cfg["bdi_threshold"])
    scid_cols = [str(c) for c in dep_cfg.get("scid_columns", [])]
    scid_pos = [str(v).strip().lower() for v in dep_cfg.get("scid_positive_values", [])]
    label_col = dep_cfg["label_column_name"]

    df = participants.copy()
    labels: list[Optional[int]] = []
    sources: list[str] = []
    excluded_reasons: list[str] = []

    has_bdi = bdi_col in df.columns
    if not has_bdi:
        logger.warning(
            "BDI column '%s' not found in participants.tsv.\n"
            "Will attempt fallback with SCID columns: %s",
            bdi_col,
            scid_cols,
        )

    for _, row in df.iterrows():
        label, source = _assign_depression_label(
            row, bdi_col, bdi_thresh, scid_cols, scid_pos, has_bdi
        )
        labels.append(label)
        sources.append(source)
        if label is None:
            excluded_reasons.append(source)

    df[label_col] = labels
    df["_label_source"] = sources

    if excluded_reasons:
        logger.info(
            "Depression labels: %d row(s) excluded. Reasons: %s",
            len(excluded_reasons),
            sorted(set(excluded_reasons)),
        )

    df = df[df[label_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)

    _log_class_distribution(df, label_col, "Depression")
    return df


def _assign_depression_label(
    row: pd.Series,
    bdi_col: str,
    bdi_thresh: float,
    scid_cols: list[str],
    scid_pos: list[str],
    has_bdi: bool,
) -> tuple[Optional[int], str]:
    """Return (label, source_description) for a single participant row."""
    # ── Primary: BDI ────────────────────────────────────────────────────
    if has_bdi:
        bdi_val = row.get(bdi_col, "")
        if _is_valid_numeric(bdi_val):
            bdi_num = float(bdi_val)
            label = 1 if bdi_num >= bdi_thresh else 0
            return label, f"BDI={bdi_num}"

    # ── Fallback: SCID columns ───────────────────────────────────────────
    for col in scid_cols:
        if col in row.index:
            val = str(row[col]).strip().lower()
            if val in ("nan", "na", "", "n/a", "none"):
                continue
            label = 1 if val in scid_pos else 0
            return label, f"SCID({col})={row[col]}"

    # ── Neither usable → exclude ─────────────────────────────────────────
    return None, "excluded:no_valid_label"


def _is_valid_numeric(val: object) -> bool:
    """Return True if val can be converted to a finite float."""
    try:
        f = float(str(val))
        return np.isfinite(f)
    except (ValueError, TypeError):
        return False


def _log_class_distribution(
    df: pd.DataFrame, label_col: str, task_name: str
) -> None:
    counts = df[label_col].value_counts().sort_index()
    logger.info(
        "%s label distribution → %s",
        task_name,
        dict(counts),
    )


# ──────────────────────────────────────────────────────────────────────────────
def balanced_subject_sample(
    subjects: list[str],
    metadata: pd.DataFrame,
    label_col: str,
    max_subjects: int,
    seed: int = 42,
) -> list[str]:
    """
    Sample *max_subjects* subjects with balanced class representation.

    This prevents the bias that occurs when participants.tsv is sorted
    by diagnosis group (all positives first, all controls last).

    Parameters
    ----------
    subjects:
        All discovered subject IDs.
    metadata:
        participants DataFrame with label column.
    label_col:
        Name of the binary label column.
    max_subjects:
        Total number of subjects to return (split evenly across classes).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of subject IDs.
    """
    rng = random.Random(seed)

    # Filter to subjects present in both the run index and metadata
    valid = [s for s in subjects if s in metadata.index]

    class_0 = [s for s in valid if metadata.loc[s, label_col] == 0]
    class_1 = [s for s in valid if metadata.loc[s, label_col] == 1]

    per_class = max_subjects // 2

    if len(class_0) < per_class or len(class_1) < per_class:
        logger.warning(
            "Balanced sampling: requested %d per class but only "
            "%d class-0 and %d class-1 subjects available. "
            "Using all available subjects from the smaller class.",
            per_class,
            len(class_0),
            len(class_1),
        )
        per_class = min(len(class_0), len(class_1))

    sampled_0 = rng.sample(class_0, per_class)
    sampled_1 = rng.sample(class_1, per_class)
    sampled = sorted(sampled_0 + sampled_1)

    logger.info(
        "Balanced sample: %d class-0, %d class-1 → %d total",
        len(sampled_0),
        len(sampled_1),
        len(sampled),
    )
    return sampled
