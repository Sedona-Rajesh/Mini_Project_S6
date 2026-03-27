from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import scipy.stats
from mne import Epochs

from eeg_dss.config import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def extract_features(
    epochs: Epochs,
    cfg: Config,
    subject_id: str = "unknown",
) -> pd.DataFrame:

    feat_cfg = cfg.features
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()
    ch_names = epochs.ch_names

    n_epochs, n_ch, n_times = data.shape

    feature_blocks: list[pd.DataFrame] = []

    # ── Statistical ─────────────────────────────────────────────
    if feat_cfg.get("statistical", True):
        feature_blocks.append(_statistical_features(data, ch_names))

    # ── Spectral ───────────────────────────────────────────────
    if feat_cfg.get("spectral", True):
        spec_df, rel_df, ratio_df = _spectral_features(
            data, ch_names, sfreq, feat_cfg
        )
        feature_blocks.extend([spec_df, rel_df, ratio_df])
        if feat_cfg.get("spectral_entropy", True):
            feature_blocks.append(
                _spectral_entropy_features(data, ch_names, sfreq, feat_cfg)
            )
        if feat_cfg.get("asymmetry", True):
            feature_blocks.append(
                _asymmetry_features(data, ch_names, sfreq, feat_cfg)
            )

    # ── Complexity ─────────────────────────────────────────────
    if feat_cfg.get("complexity", True):
        feature_blocks.append(_complexity_features(data, ch_names, sfreq))

    # ── 🔥 NEW: Advanced depression features ───────────────────
    feature_blocks.append(_advanced_depression_features(data, ch_names))

    # ── Connectivity (optional) ────────────────────────────────
    if feat_cfg.get("connectivity", False):
        feature_blocks.append(
            _connectivity_features(data, ch_names, sfreq, feat_cfg)
        )

    features = pd.concat(feature_blocks, axis=1)
    features = _sanitize(features, feat_cfg)

    features.insert(0, "subject_id", subject_id)
    return features


# ─────────────────────────────────────────────────────────────
def _statistical_features(data, ch_names):
    records = []
    for ep in data:
        row = {}
        for i, ch in enumerate(ch_names):
            x = ep[i]
            row[f"{ch}_mean"] = np.mean(x)
            row[f"{ch}_std"] = np.std(x)
            row[f"{ch}_skew"] = scipy.stats.skew(x)
            row[f"{ch}_kurt"] = scipy.stats.kurtosis(x)
            row[f"{ch}_ptp"] = np.ptp(x)
        records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
def _spectral_features(data, ch_names, sfreq, feat_cfg):

    bands = feat_cfg.get("bands")
    n_fft = feat_cfg.get("psd_n_fft") or min(256, data.shape[-1])
    ratios = feat_cfg.get("power_ratios", [])

    abs_records, rel_records, ratio_records = [], [], []

    for ep in data:
        abs_row, rel_row, ratio_row = {}, {}, {}

        for i, ch in enumerate(ch_names):
            freqs, psd = _welch_psd(ep[i], sfreq, n_fft)

            band_powers = {}
            for b, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                power = np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0
                band_powers[b] = power
                abs_row[f"{ch}_abs_{b}"] = power

            total = sum(band_powers.values()) + 1e-12
            for b, p in band_powers.items():
                rel_row[f"{ch}_rel_{b}"] = p / total

        for a, b in ratios:
            a_vals = [abs_row.get(f"{ch}_abs_{a}", 0) for ch in ch_names]
            b_vals = [abs_row.get(f"{ch}_abs_{b}", 0) for ch in ch_names]
            ratio_row[f"ratio_{a}_over_{b}"] = np.mean(a_vals) / (
                np.mean(b_vals) + 1e-12
            )

        abs_records.append(abs_row)
        rel_records.append(rel_row)
        ratio_records.append(ratio_row)

    return (
        pd.DataFrame(abs_records),
        pd.DataFrame(rel_records),
        pd.DataFrame(ratio_records),
    )


def _welch_psd(x, sfreq, n_fft):
    from scipy.signal import welch

    return welch(x, fs=sfreq, nperseg=min(n_fft, len(x)))


# ─────────────────────────────────────────────────────────────
def _complexity_features(data, ch_names, sfreq):
    records = []
    for ep in data:
        row = {}
        for i, ch in enumerate(ch_names):
            x = ep[i]
            activity = np.var(x)
            dx = np.diff(x)
            ddx = np.diff(dx)

            mobility = np.sqrt(np.var(dx) / (activity + 1e-12))
            complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (
                mobility + 1e-12
            )

            row[f"{ch}_hjorth_activity"] = activity
            row[f"{ch}_hjorth_mobility"] = mobility
            row[f"{ch}_hjorth_complexity"] = complexity

        records.append(row)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 🔥 NEW FEATURES
def _advanced_depression_features(data, ch_names):
    records = []
    ch_index = {ch: i for i, ch in enumerate(ch_names)}

    for ep in data:
        row = {}

        # Frontal asymmetry
        if "F3" in ch_index and "F4" in ch_index:
            f3 = ep[ch_index["F3"]]
            f4 = ep[ch_index["F4"]]

            diff = f3 - f4
            row["frontal_asymmetry_mean"] = np.mean(diff)
            row["frontal_asymmetry_std"] = np.std(diff)

        # Low vs high energy ratio
        n = ep.shape[1]
        low = ep[:, : n // 3]
        high = ep[:, -n // 3 :]

        low_power = np.mean(low**2)
        high_power = np.mean(high**2)

        row["low_high_energy_ratio"] = (
            low_power / (high_power + 1e-12)
        )

        records.append(row)

    return pd.DataFrame(records)


def _spectral_entropy_features(data, ch_names, sfreq, feat_cfg):
    bands = feat_cfg.get("bands", {})
    n_fft = feat_cfg.get("psd_n_fft") or min(256, data.shape[-1])
    eps = 1e-12

    records = []
    for ep in data:
        row = {}
        for i, ch in enumerate(ch_names):
            freqs, psd = _welch_psd(ep[i], sfreq, n_fft)
            psd = np.maximum(psd, 0)

            psd_sum = np.sum(psd) + eps
            p = psd / psd_sum
            row[f"{ch}_spec_entropy"] = float(
                -np.sum(p * np.log2(p + eps)) / np.log2(len(p) + eps)
            )

            for b, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                if not np.any(idx):
                    row[f"{ch}_{b}_spec_entropy"] = 0.0
                    continue
                b_psd = np.maximum(psd[idx], 0)
                b_p = b_psd / (np.sum(b_psd) + eps)
                row[f"{ch}_{b}_spec_entropy"] = float(
                    -np.sum(b_p * np.log2(b_p + eps))
                    / np.log2(len(b_p) + eps)
                )
        records.append(row)

    return pd.DataFrame(records)


def _asymmetry_features(data, ch_names, sfreq, feat_cfg):
    pairs = feat_cfg.get(
        "asymmetry_pairs",
        [["F3", "F4"], ["C3", "C4"], ["P3", "P4"], ["O1", "O2"]],
    )
    bands = feat_cfg.get("bands", {})
    asym_bands = feat_cfg.get("asymmetry_bands", ["theta", "alpha", "beta"])
    n_fft = feat_cfg.get("psd_n_fft") or min(256, data.shape[-1])
    ch_index = {ch: i for i, ch in enumerate(ch_names)}
    eps = 1e-12

    valid_pairs = [
        (left, right) for left, right in pairs if left in ch_index and right in ch_index
    ]
    if not valid_pairs:
        return pd.DataFrame(index=range(data.shape[0]))

    records = []
    for ep in data:
        row = {}
        rel_by_ch: dict[str, dict[str, float]] = {}

        for ch in {c for pair in valid_pairs for c in pair}:
            freqs, psd = _welch_psd(ep[ch_index[ch]], sfreq, n_fft)
            band_powers = {}
            for b, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                band_powers[b] = (
                    np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0
                )
            total = sum(band_powers.values()) + eps
            rel_by_ch[ch] = {b: p / total for b, p in band_powers.items()}

        for left, right in valid_pairs:
            for b in asym_bands:
                if b not in rel_by_ch[left] or b not in rel_by_ch[right]:
                    continue
                l_val = rel_by_ch[left][b]
                r_val = rel_by_ch[right][b]
                row[f"asym_{left}_{right}_{b}_rel_diff"] = l_val - r_val
                row[f"asym_{left}_{right}_{b}_rel_logratio"] = np.log(
                    (l_val + eps) / (r_val + eps)
                )
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
def _sanitize(df, feat_cfg):
    df = df.replace([np.inf, -np.inf], np.nan)
    max_nan_fraction = float(feat_cfg.get("max_nan_fraction", 0.4))
    clip_quantile = float(feat_cfg.get("clip_quantile", 0.001))

    for col in df.columns:
        col_series = pd.to_numeric(df[col], errors="coerce")
        nan_fraction = float(col_series.isna().mean())
        if nan_fraction >= 1.0:
            df[col] = 0.0
            continue

        if nan_fraction > max_nan_fraction:
            logger.debug(
                "Feature '%s' has %.1f%% NaN values; filling with median.",
                col,
                100.0 * nan_fraction,
            )

        median = float(col_series.median()) if not np.isnan(col_series.median()) else 0.0
        col_series = col_series.fillna(median)

        if 0.0 < clip_quantile < 0.5:
            lo = col_series.quantile(clip_quantile)
            hi = col_series.quantile(1.0 - clip_quantile)
            col_series = col_series.clip(lo, hi)

        df[col] = col_series

    return df


# ─────────────────────────────────────────────────────────────
def _connectivity_features(data, ch_names, sfreq, feat_cfg):
    return pd.DataFrame()  # unchanged placeholder

# ─────────────────────────────────────────────────────────────
def validate_feature_schema(df, expected_columns):
    """
    Check that df contains expected feature columns.
    """
    df_cols = set(df.columns) - {"subject_id", "label"}
    expected_set = set(expected_columns)

    missing = sorted(expected_set - df_cols)
    extra = sorted(df_cols - expected_set)

    if missing or extra:
        raise ValueError(
            f"Feature schema mismatch.\n"
            f"Missing: {missing[:5]}\n"
            f"Extra: {extra[:5]}"
        )
        