"""
eeg_dss/data/loader.py
──────────────────────
BIDS-aware EEG discovery and loading.

Responsibilities
----------------
* Dynamically discover all EEG runs under a BIDS root (no hard-coded IDs).
* Load raw MNE objects from .set, .fif, .edf, .bdf, .vhdr files.
* Harmonize channel names toward a target montage when possible.
* Return structured DataFrames for downstream processing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import mne
import pandas as pd

logger = logging.getLogger(__name__)

# BIDS EEG file extensions supported (in preference order)
_EEG_EXTENSIONS = [".set", ".fif", ".edf", ".bdf", ".vhdr", ".cnt"]

# Regex for BIDS subject directory names
_SUB_RE = re.compile(r"^sub-(.+)$")


# ──────────────────────────────────────────────────────────────────────────────
def discover_bids_subjects(bids_root: Path) -> list[str]:
    """
    Return sorted list of subject IDs found under *bids_root*.

    Parameters
    ----------
    bids_root:
        Root of the BIDS dataset (contains sub-* directories).

    Returns
    -------
    List of subject ID strings (e.g. ["001", "002", ...]).
    """
    if not bids_root.exists():
        raise FileNotFoundError(
            f"BIDS root not found: {bids_root}\n"
            "Check data.raw_root and the dataset folder name in config.yaml."
        )

    subjects = []
    for d in sorted(bids_root.iterdir()):
        m = _SUB_RE.match(d.name)
        if m and d.is_dir():
            subjects.append(m.group(1))

    if not subjects:
        raise RuntimeError(
            f"No sub-* directories found under {bids_root}.\n"
            "Verify this is a valid BIDS dataset with at least one subject."
        )

    logger.info("Discovered %d subjects under %s", len(subjects), bids_root)
    return subjects


# ──────────────────────────────────────────────────────────────────────────────
def find_eeg_files(bids_root: Path, subject_id: str) -> list[Path]:
    """
    Return all EEG files for a single subject (across sessions and tasks).

    Searches recursively under sub-<subject_id>/eeg/.
    """
    sub_dir = bids_root / f"sub-{subject_id}"
    if not sub_dir.exists():
        logger.warning("Subject directory missing: %s", sub_dir)
        return []

    files: list[Path] = []
    for ext in _EEG_EXTENSIONS:
        files.extend(sub_dir.rglob(f"*{ext}"))

    # Deduplicate (some formats have sidecar files)
    unique: list[Path] = []
    seen: set[str] = set()
    for f in sorted(files):
        key = f.stem.split(".")[0]  # strip double extensions
        if key not in seen:
            seen.add(key)
            unique.append(f)

    logger.debug(
        "Subject %s: found %d EEG file(s)", subject_id, len(unique)
    )
    return unique


# ──────────────────────────────────────────────────────────────────────────────
def load_raw(eeg_path: Path, preload: bool = True) -> mne.io.BaseRaw:
    """
    Load an EEG file into an MNE Raw object.

    Supports .set (EEGLAB), .fif (MNE), .edf, .bdf, .vhdr (BrainVision),
    and .cnt (ANT Neuroscan).

    Parameters
    ----------
    eeg_path:
        Absolute path to the EEG file.
    preload:
        Whether to preload data into memory.

    Returns
    -------
    mne.io.BaseRaw
    """
    ext = eeg_path.suffix.lower()
    logger.debug("Loading %s", eeg_path)

    loaders = {
        ".set": lambda p: mne.io.read_raw_eeglab(p, preload=preload, verbose=False),
        ".fif": lambda p: mne.io.read_raw_fif(p, preload=preload, verbose=False),
        ".edf": lambda p: mne.io.read_raw_edf(p, preload=preload, verbose=False),
        ".bdf": lambda p: mne.io.read_raw_bdf(p, preload=preload, verbose=False),
        ".vhdr": lambda p: mne.io.read_raw_brainvision(p, preload=preload, verbose=False),
        ".cnt": lambda p: mne.io.read_raw_cnt(p, preload=preload, verbose=False),
    }

    if ext not in loaders:
        raise ValueError(
            f"Unsupported EEG format '{ext}' for file: {eeg_path}\n"
            f"Supported formats: {list(loaders)}"
        )

    try:
        raw = loaders[ext](eeg_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load EEG file: {eeg_path}\nUnderlying error: {exc}"
        ) from exc

    logger.debug(
        "Loaded %s — %d ch, %.1f s @ %.0f Hz",
        eeg_path.name,
        len(raw.ch_names),
        raw.times[-1],
        raw.info["sfreq"],
    )
    return raw


# ──────────────────────────────────────────────────────────────────────────────
def harmonize_channels(
    raw: mne.io.BaseRaw,
    target_montage_name: str = "standard_1020",
    required_channels: list[str] | None = None,
) -> mne.io.BaseRaw:
    """
    Attempt to reconcile channel names with a target montage.

    Strategy
    --------
    1. Uppercase all channel names.
    2. Try to set the target montage; channels without known positions are
       dropped rather than crashing.
    3. If ``required_channels`` is set, verify they are all present after
       harmonization.

    Parameters
    ----------
    raw:
        MNE Raw object (modified in-place copy returned).
    target_montage_name:
        Name understood by ``mne.channels.make_standard_montage``.
    required_channels:
        If provided, raise if any of these are missing after harmonization.

    Returns
    -------
    mne.io.BaseRaw  (copy with harmonized channels)
    """
    raw = raw.copy()

    # --- Rename channels to uppercase -----------------------------------
    rename_map = {ch: ch.upper() for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    # --- Set montage, drop channels with no known position --------------
    try:
        montage = mne.channels.make_standard_montage(target_montage_name)
        known = set(montage.ch_names)
        unknown = [ch for ch in raw.ch_names if ch not in known]
        if unknown:
            logger.debug(
                "Dropping %d channel(s) not in montage %s: %s",
                len(unknown),
                target_montage_name,
                unknown[:10],
            )
            raw.drop_channels(unknown)

        raw.set_montage(montage, on_missing="ignore", verbose=False)

    except Exception as exc:
        logger.warning(
            "Could not set montage '%s': %s. Proceeding without montage.",
            target_montage_name,
            exc,
        )

    # --- Check required channels ----------------------------------------
    if required_channels:
        missing = [c for c in required_channels if c not in raw.ch_names]
        if missing:
            raise RuntimeError(
                f"Required channels missing after harmonization: {missing}\n"
                f"Available channels: {raw.ch_names[:20]}"
            )

    return raw


# ──────────────────────────────────────────────────────────────────────────────
def build_run_index(
    bids_root: Path,
    subjects: list[str],
) -> pd.DataFrame:
    """
    Build a DataFrame index of all EEG runs across subjects.

    Columns: subject_id, file_path
    """
    records = []
    for sub in subjects:
        files = find_eeg_files(bids_root, sub)
        if not files:
            logger.warning("No EEG files found for subject %s — skipping", sub)
            continue
        for f in files:
            records.append({"subject_id": sub, "file_path": str(f)})

    if not records:
        raise RuntimeError(
            f"No EEG files found for any subject under {bids_root}.\n"
            "Check the BIDS structure: expected sub-*/eeg/*.<ext> layout."
        )

    df = pd.DataFrame(records)
    logger.info(
        "Run index built: %d runs across %d subjects",
        len(df),
        df["subject_id"].nunique(),
    )
    return df
