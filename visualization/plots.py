"""
eeg_dss/visualization/plots.py
────────────────────────────────
Standalone plotting utilities used by both the evaluation pipeline
and the Streamlit app.

All functions write to disk when ``out_dir`` is provided, and always
return the matplotlib Figure so callers can embed it directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def plot_epoch_probability_histogram(
    epoch_probabilities: list[float],
    threshold: float,
    task: str,
    out_dir: Optional[Path] = None,
):
    """
    Histogram of per-epoch positive-class probabilities with threshold line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(epoch_probabilities, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Probability (positive class)")
    ax.set_ylabel("Epoch count")
    ax.set_title(f"{task.capitalize()} — Epoch Probability Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if out_dir:
        path = Path(out_dir) / f"{task}_epoch_proba_hist.png"
        fig.savefig(path, dpi=150)
        logger.info("Epoch probability histogram saved: %s", path)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    task: str,
    out_dir: Optional[Path] = None,
    n_bins: int = 10,
):
    """Reliability / calibration curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if len(np.unique(y_true)) < 2:
        logger.warning("Calibration curve skipped: only one class present.")
        return None

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "s-", color="#4C72B0", label=task.capitalize())
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{task.capitalize()} — Calibration Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if out_dir:
        path = Path(out_dir) / f"{task}_calibration_curve.png"
        fig.savefig(path, dpi=150)
        logger.info("Calibration curve saved: %s", path)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_subject_probability_bar(
    subject_df: pd.DataFrame,
    threshold: float,
    task: str,
    out_dir: Optional[Path] = None,
):
    """
    Horizontal bar chart of mean subject-level probabilities, coloured
    by true label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = subject_df.sort_values("mean_proba", ascending=True).reset_index(drop=True)
    colours = ["#4C72B0" if lbl == 0 else "#DD4444" for lbl in df["true_label"]]

    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.35)))
    ax.barh(df["subject_id"], df["mean_proba"], color=colours, edgecolor="none")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"Threshold={threshold:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean positive-class probability")
    ax.set_title(f"{task.capitalize()} — Subject-Level Predictions")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Control (true)"),
        Patch(facecolor="#DD4444", label="Positive (true)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()

    if out_dir:
        path = Path(out_dir) / f"{task}_subject_proba_bars.png"
        fig.savefig(path, dpi=150)
        logger.info("Subject probability bar chart saved: %s", path)

    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_raw_psd(raw, task: str, max_channels: int = 10):
    """
    Plot power spectral density of an MNE Raw object (used in Streamlit).
    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = raw.ch_names[:max_channels]
    fig, ax = plt.subplots(figsize=(7, 3))
    raw.compute_psd(picks=picks, fmax=50, verbose=False).plot(
        axes=ax, show=False, picks=picks
    )
    ax.set_title(f"PSD — first {len(picks)} channels")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_electrode_positions(raw, title: str = "EEG Electrode Positions"):
    """
    Plot scalp electrode positions from channel montage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111)
    raw.plot_sensors(
        kind="topomap",
        ch_type="eeg",
        show_names=True,
        axes=ax,
        show=False,
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_band_topomaps(raw, bands: dict[str, list[float]], max_bands: int = 4):
    """
    Plot EEG scalp topomaps for configured frequency bands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    if not bands:
        raise ValueError("No frequency bands configured for topomap.")

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise ValueError("No EEG channels available for topomap plotting.")

    selected_bands = list(bands.items())[:max_bands]
    psd = raw.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=45.0,
        picks=picks,
        verbose=False,
    )
    freqs = psd.freqs
    psd_data = psd.get_data()  # shape: (n_channels, n_freqs)

    n_cols = min(2, len(selected_bands))
    n_rows = int(np.ceil(len(selected_bands) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (band_name, band_range) in enumerate(selected_bands):
        lo, hi = float(band_range[0]), float(band_range[1])
        band_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(band_mask):
            band_power = np.zeros(len(picks))
        else:
            band_power = np.trapz(psd_data[:, band_mask], freqs[band_mask], axis=1)

        mne.viz.plot_topomap(
            band_power,
            raw.info,
            picks=picks,
            axes=axes[idx],
            show=False,
            contours=6,
            cmap="RdBu_r",
        )
        axes[idx].set_title(f"{band_name.capitalize()} ({lo:.1f}-{hi:.1f} Hz)", fontsize=10)

    for j in range(len(selected_bands), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Brain Topomaps (Band Power)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_confidence_gauge(probability: float, threshold: float, task: str):
    """
    Simple gauge-like matplotlib figure for the Streamlit app.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis("off")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#DDDDDD", linewidth=18,
            solid_capstyle="round")

    # Filled arc up to probability
    theta_fill = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    colour = "#DD4444" if probability >= threshold else "#4C72B0"
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=colour, linewidth=18,
            solid_capstyle="round")

    # Threshold tick
    t_angle = np.pi - threshold * np.pi
    ax.plot(
        [0.72 * np.cos(t_angle), 0.92 * np.cos(t_angle)],
        [0.72 * np.sin(t_angle), 0.92 * np.sin(t_angle)],
        color="black", linewidth=2,
    )

    # Centre text
    label = "POSITIVE" if probability >= threshold else "CONTROL"
    ax.text(0, 0.15, f"{probability:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=colour)
    ax.text(0, -0.05, label, ha="center", va="center",
            fontsize=11, color=colour)
    ax.set_title(f"{task.capitalize()} Probability", fontsize=10, pad=2)

    fig.tight_layout()
    return fig


def plot_electrode_positions(ch_names: list[str], montage_name: str = "standard_1020"):
    """Plot available electrode positions on a scalp layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    known = [ch for ch in ch_names if ch in montage.ch_names]

    if len(known) < 3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough known EEG channels for scalp map", ha="center", va="center")
        return fig

    info = mne.create_info(ch_names=known, sfreq=256.0, ch_types="eeg")
    info.set_montage(montage)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax, show=False)
    ax.set_title("Electrode Positions")
    fig.tight_layout()
    return fig


def plot_scalp_topomap(
    ch_names: list[str],
    values: list[float],
    title: str,
    montage_name: str = "standard_1020",
    cmap: str = "RdBu_r",
):
    """Plot a scalp topomap from channel-aligned values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    coords = montage.get_positions()["ch_pos"]

    use_names = []
    use_vals = []
    for ch, val in zip(ch_names, values):
        if ch in coords:
            use_names.append(ch)
            use_vals.append(float(val))

    if len(use_names) < 3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough known EEG channels for topomap", ha="center", va="center")
        return fig

    pos = np.asarray([[coords[ch][0], coords[ch][1]] for ch in use_names], dtype=float)
    vals = np.asarray(use_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(vals, pos, axes=ax, show=False, cmap=cmap, contours=0)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Normalized intensity", rotation=90)
    fig.tight_layout()
    return fig
