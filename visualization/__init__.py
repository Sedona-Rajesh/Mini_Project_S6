"""Visualization sub-package."""
from .plots import (
    plot_calibration_curve,
    plot_confidence_gauge,
    plot_electrode_positions,
    plot_epoch_probability_histogram,
    plot_raw_psd,
    plot_scalp_topomap,
    plot_subject_probability_bar,
)

# Backward-compatible alias
plot_subject_predictions_bar = plot_subject_probability_bar

__all__ = [
    "plot_calibration_curve",
    "plot_confidence_gauge",
    "plot_electrode_positions",
    "plot_epoch_probability_histogram",
    "plot_raw_psd",
    "plot_scalp_topomap",
    "plot_subject_probability_bar",
    "plot_subject_predictions_bar",
]
