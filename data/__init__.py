"""Data loading and metadata sub-package."""
from .loader import (
    build_run_index,
    discover_bids_subjects,
    find_eeg_files,
    harmonize_channels,
    load_raw,
)
from .metadata import (
    balanced_subject_sample,
    infer_alzheimer_labels,
    infer_depression_labels,
    load_participants_tsv,
)

__all__ = [
    "build_run_index",
    "discover_bids_subjects",
    "find_eeg_files",
    "harmonize_channels",
    "load_raw",
    "balanced_subject_sample",
    "infer_alzheimer_labels",
    "infer_depression_labels",
    "load_participants_tsv",
]
