"""
eeg_dss/preprocessing/pipeline.py
───────────────────────────────────
Full preprocessing pipeline: filter → bad-channel detection → optional ICA
→ re-reference → epoch → reject → return clean Epochs.

All parameters are read from the Config object — zero magic constants.
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import numpy as np

from eeg_dss.config import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def preprocess_raw(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """
    Apply bandpass + notch filtering, bad-channel detection/interpolation,
    optional ICA, and re-referencing to a Raw object.

    Parameters
    ----------
    raw:
        Loaded (and channel-harmonized) MNE Raw object. Modified in-place.
    cfg:
        Loaded Config.

    Returns
    -------
    mne.io.BaseRaw  (filtered, clean)
    """
    pp = cfg.preprocessing

    # ── 1. Pick only EEG channels ────────────────────────────────────────
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise RuntimeError(
            "No EEG channels found in the raw data.\n"
            "Check channel types in the raw file and harmonize_channels output."
        )

    raw = raw.copy()
    raw.pick_types(eeg=True, verbose=False)

    min_eeg_channels = int(pp.get("min_eeg_channels", 1))
    if len(raw.ch_names) < min_eeg_channels:
        raise RuntimeError(
            f"Only {len(raw.ch_names)} EEG channels remain after harmonization; "
            f"minimum required is {min_eeg_channels}."
        )

    # ── 2. Resample ──────────────────────────────────────────────────────
    target_sfreq = float(pp["target_sfreq"])
    if raw.info["sfreq"] != target_sfreq:
        logger.debug("Resampling %.1f → %.1f Hz", raw.info["sfreq"], target_sfreq)
        raw.resample(target_sfreq, verbose=False)

    # ── 3. Bandpass filter ───────────────────────────────────────────────
    l_freq = pp["bandpass"]["l_freq"]
    h_freq = pp["bandpass"]["h_freq"]
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )
    logger.debug("Bandpass filter: %.1f – %.1f Hz", l_freq, h_freq)

    # ── 4. Notch filter ──────────────────────────────────────────────────
    notch_freqs = pp.get("notch_freqs", [])
    if notch_freqs:
        # only apply notch frequencies that are below the bandpass high
        nyquist = raw.info["sfreq"] / 2.0
        applicable = [f for f in notch_freqs if f <= h_freq and f < nyquist]
        if applicable:
            raw.notch_filter(freqs=applicable, verbose=False)
            logger.debug("Notch filter applied: %s Hz", applicable)

    # ── 5. Bad channel detection ─────────────────────────────────────────
    method = pp.get("bad_channel_method", "none")
    if method != "none":
        raw = _detect_bad_channels(raw, method, pp)

    # ── 6. Interpolate bad channels ──────────────────────────────────────
    if pp.get("interpolate_bad_channels", False) and raw.info["bads"]:
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
            logger.debug("Interpolated bad channels")
        except Exception as exc:
            logger.warning("Channel interpolation failed: %s", exc)

    # ── 7. ICA artifact removal (optional) ──────────────────────────────
    if pp.get("run_ica", False):
        raw = _apply_ica(raw, cfg)

    # ── 8. Re-reference ──────────────────────────────────────────────────
    ref = pp.get("reference", "average")
    if ref == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    else:
        if ref in raw.ch_names:
            raw.set_eeg_reference([ref], projection=False, verbose=False)
        else:
            logger.warning(
                "Reference channel '%s' not found; using average reference.", ref
            )
            raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw


# ──────────────────────────────────────────────────────────────────────────────
def make_epochs(
    raw: mne.io.BaseRaw, cfg: Config
) -> Optional[mne.Epochs]:
    """
    Segment cleaned Raw into fixed-length epochs.

    Parameters
    ----------
    raw:
        Cleaned MNE Raw object.
    cfg:
        Loaded Config.

    Returns
    -------
    mne.Epochs or None if no valid epochs survive rejection.
    """
    pp = cfg.preprocessing
    duration = float(pp["epoch_duration_sec"])
    overlap = float(pp.get("epoch_overlap_sec", 0.0))
    reject_uv = pp.get("reject_peak_to_peak_uv", None)
    flat_uv = pp.get("reject_flat_uv", 0.01)
    baseline = pp.get("baseline", None)
    baseline = tuple(baseline) if baseline else None

    reject = None
    if reject_uv is not None:
        reject = {"eeg": float(reject_uv) * 1e-6}  # µV → V

    events = mne.make_fixed_length_events(
        raw, duration=duration, overlap=overlap
    )

    if len(events) == 0:
        logger.warning(
            "No events created from fixed-length epochs "
            "(recording too short? duration=%.1f s)", duration
        )
        return None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=0.0,
        tmax=duration,
        baseline=baseline,
        reject=reject,
        flat={"eeg": float(flat_uv) * 1e-6},
        preload=True,
        verbose=False,
    )

    n_in = len(events)
    n_out = len(epochs)
    logger.debug(
        "Epochs: %d created, %d survived rejection (%.0f%%)",
        n_in,
        n_out,
        100.0 * n_out / n_in if n_in else 0,
    )

    if n_out == 0:
        logger.warning(
            "All epochs rejected (threshold=%.1f µV). "
            "Consider raising reject_peak_to_peak_uv in config.",
            reject_uv,
        )
        return None

    return epochs


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────
def _detect_bad_channels(
    raw: mne.io.BaseRaw,
    method: str,
    pp: dict,
) -> mne.io.BaseRaw:
    """Detect and mark bad channels using the specified method."""
    if method == "flat":
        # Simple flat-channel detection: std < 0.1 µV
        data = raw.get_data(picks="eeg") * 1e6  # → µV
        stds = data.std(axis=1)
        flat_thresh = float(pp.get("bad_channel_flat_std_uv", 0.1))
        bads = [
            raw.ch_names[i]
            for i, s in enumerate(stds)
            if s < flat_thresh
        ]
        if bads:
            raw.info["bads"].extend(bads)
            logger.debug("Flat channels detected: %s", bads)

    elif method == "ransac":
        try:
            from pyprep.find_noisy_channels import NoisyChannels

            nd = NoisyChannels(raw, random_state=42)
            nd.find_all_bads(ransac=True)
            raw.info["bads"] = list(set(raw.info["bads"]) | set(nd.get_bads()))
            logger.debug("RANSAC bad channels: %s", nd.get_bads())
        except ImportError:
            logger.warning(
                "pyprep not installed; falling back to flat-channel detection. "
                "Install with: pip install pyprep"
            )
            return _detect_bad_channels(raw, "flat", pp)
        except Exception as exc:
            logger.warning("RANSAC failed: %s — skipping bad channel detection.", exc)

    else:
        logger.warning("Unknown bad_channel_method '%s'; skipping.", method)

    return raw


def _apply_ica(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Run ICA and auto-remove EOG/ECG components."""
    pp = cfg.preprocessing
    n_components = int(pp.get("ica_n_components", 20))
    method = pp.get("ica_method", "fastica")

    logger.info("Running ICA (%s, n=%d)…", method, n_components)

    n_channels = len(mne.pick_types(raw.info, eeg=True))
    n_components = min(n_components, n_channels - 1)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=cfg.seed,
        max_iter="auto",
        verbose=False,
    )

    try:
        ica.fit(raw, verbose=False)
    except Exception as exc:
        logger.warning("ICA fit failed: %s — skipping ICA.", exc)
        return raw

    # Auto-label EOG and ECG components
    exclude_idx: list[int] = []
    for artifact_type in ["eog", "ecg"]:
        try:
            idx, _ = ica.find_bads_eog(raw) if artifact_type == "eog" else ica.find_bads_ecg(raw)
            exclude_idx.extend(idx)
        except Exception:
            pass

    if exclude_idx:
        ica.exclude = list(set(exclude_idx))
        logger.debug("ICA excluding components: %s", ica.exclude)

    ica.apply(raw, verbose=False)
    return raw
