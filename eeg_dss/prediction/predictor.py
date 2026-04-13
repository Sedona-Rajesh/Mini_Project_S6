"""
eeg_dss/prediction/predictor.py
────────────────────────────────
Run inference on a new EEG file using a saved model artifact.

Steps
-----
1. Load model artifact (model, scaler, feature_names, threshold).
2. Load and preprocess the input EEG file.
3. Extract features with the same pipeline used during training.
4. Align features to the saved feature schema.
5. Return per-epoch and aggregate predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_dss.config import Config
from eeg_dss.data.loader import harmonize_channels, load_raw
from eeg_dss.features.extractor import extract_features
from eeg_dss.preprocessing.pipeline import make_epochs, preprocess_raw
from eeg_dss.training.trainer import load_model_artifact

logger = logging.getLogger(__name__)
_TRAPZ = np.trapz if hasattr(np, "trapz") else np.trapezoid


# ──────────────────────────────────────────────────────────────────────────────
def predict_from_file(
    eeg_path: str | Path,
    task: str,
    cfg: Config,
) -> dict[str, Any]:
    """
    Run the full inference pipeline on a single EEG file.

    Parameters
    ----------
    eeg_path:
        Path to the EEG file (.set, .fif, .edf, .bdf, .vhdr).
    task:
        ``"alzheimer"`` or ``"depression"``.
    cfg:
        Loaded Config.

    Returns
    -------
    dict with keys:
        - ``predicted_label``  (int 0/1)
        - ``probability``      (float, probability of positive class)
        - ``n_epochs``         (int)
        - ``epoch_probabilities``  (list[float])
    """
    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")

    raw_clean, epochs, feature_df = _load_preprocessed_feature_frame(eeg_path, cfg)
    task_result = _predict_task_from_features(task, cfg, feature_df)

    result = {
        "predicted_label": task_result["predicted_label"],
        "predicted_class": task_result["predicted_class"],
        "probability": task_result["probability"],
        "n_epochs": int(len(feature_df)),
        "epoch_probabilities": task_result["epoch_probabilities"],
        "inference_mode": task_result["inference_mode"],
    }

    logger.info(
        "Inference [%s]: %d epochs → label=%d (%s), prob=%.3f, confidence=%.1f%%",
        task,
        len(task_result["epoch_probabilities"]),
        result["predicted_label"],
        result["predicted_class"],
        result["probability"],
        abs(result["probability"] - 0.5) * 200,
    )

    return result


def predict_dual_from_file(
    eeg_path: str | Path,
    cfg: Config,
) -> dict[str, Any]:
    """
    Run one EEG file through both Alzheimer and Depression models and
    return a triage decision.
    """
    raw_clean, epochs, feature_df = _load_preprocessed_feature_frame(eeg_path, cfg)

    alz = _predict_task_from_features("alzheimer", cfg, feature_df)
    dep = _predict_task_from_features("depression", cfg, feature_df)

    band_maps = _compute_band_power_maps(epochs, cfg)
    domain_maps = _compute_domain_evidence_maps(band_maps)
    topomap_analysis = _build_topomap_analysis(epochs.ch_names, band_maps, domain_maps)

    alz_domain = _safe_mean(domain_maps.get("alzheimer", []))
    dep_domain = _safe_mean(domain_maps.get("depression", []))

    alz_score = float(alz["model_margin"] + 0.15 * alz_domain)
    dep_score = float(dep["model_margin"] + 0.15 * dep_domain)

    triage_cfg = cfg.get("prediction", "triage", default={}) or {}
    min_evidence = float(triage_cfg.get("min_evidence", 0.05))
    min_gap = float(triage_cfg.get("min_separation", 0.08))
    min_model_margin = float(triage_cfg.get("min_model_margin", 0.06))
    min_alz_margin = float(triage_cfg.get("min_alzheimer_margin", min_model_margin))
    min_dep_margin = float(triage_cfg.get("min_depression_margin", min_model_margin))
    control_label = str(triage_cfg.get("control_label", "Healthy"))

    decision = _triage_decision(
        alz_score,
        dep_score,
        min_evidence,
        min_gap,
        alz["model_margin"],
        dep["model_margin"],
        alz["predicted_label"],
        dep["predicted_label"],
        min_alz_margin,
        min_dep_margin,
        control_label,
    )
    medical_interpretation = _build_medical_interpretation(
        topomap_analysis,
        decision,
        alz_score,
        dep_score,
    )
    clinical_recommendations = _build_clinical_recommendations(
        topomap_analysis,
        decision,
        alz_score,
        dep_score,
    )

    return {
        "final_prediction": decision["label"],
        "decision_reason": decision["reason"],
        "medical_interpretation": medical_interpretation,
        "clinical_recommendations": clinical_recommendations,
        "n_epochs": int(len(feature_df)),
        "alzheimer": {
            **_public_task_payload(alz),
            "domain_signal": alz_domain,
            "evidence_score": alz_score,
        },
        "depression": {
            **_public_task_payload(dep),
            "domain_signal": dep_domain,
            "evidence_score": dep_score,
        },
        "scalp": {
            "channels": epochs.ch_names,
            "band_power_maps": band_maps,
            "domain_evidence_maps": domain_maps,
            "analysis": topomap_analysis,
        },
    }


def _public_task_payload(task_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_label": task_result["predicted_label"],
        "predicted_class": task_result["predicted_class"],
        "probability": task_result["probability"],
        "inference_mode": task_result["inference_mode"],
        "model_margin": task_result["model_margin"],
    }


def _triage_decision(
    alz_score: float,
    dep_score: float,
    min_evidence: float,
    min_gap: float,
    alz_margin: float,
    dep_margin: float,
    alz_label: int,
    dep_label: int,
    min_alz_margin: float,
    min_dep_margin: float,
    control_label: str,
):
    top = max(alz_score, dep_score)
    gap = abs(alz_score - dep_score)

    if int(alz_label) == 0 and int(dep_label) == 0:
        return {
            "label": control_label,
            "reason": "Both disease-specific models are below their diagnostic thresholds.",
        }

    if alz_margin < min_alz_margin and dep_margin < min_dep_margin:
        return {
            "label": control_label,
            "reason": "Both disease model margins are below minimum disease confidence.",
        }

    if top < min_evidence:
        return {
            "label": "Inconclusive",
            "reason": "Both disease evidence scores are below minimum confidence.",
        }

    if gap < min_gap:
        return {
            "label": "Inconclusive",
            "reason": "Alzheimer and Depression evidence are too close to separate reliably.",
        }

    if alz_score > dep_score:
        if alz_margin < min_alz_margin:
            return {
                "label": "Inconclusive",
                "reason": "Alzheimer evidence score is higher, but model margin is below reliability gate.",
            }
        return {
            "label": "Alzheimer",
            "reason": "Alzheimer evidence exceeded Depression evidence by configured margin.",
        }
    if dep_margin < min_dep_margin:
        return {
            "label": "Inconclusive",
            "reason": "Depression evidence score is higher, but model margin is below reliability gate.",
        }
    return {
        "label": "Depression",
        "reason": "Depression evidence exceeded Alzheimer evidence by configured margin.",
    }


def _load_preprocessed_feature_frame(eeg_path: str | Path, cfg: Config):
    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")

    raw = load_raw(eeg_path)
    raw = harmonize_channels(
        raw,
        target_montage_name=cfg.montage.get("target", "standard_1020"),
        required_channels=cfg.montage.get("required_channels") or None,
    )
    raw_clean = preprocess_raw(raw, cfg)
    epochs = make_epochs(raw_clean, cfg)

    if epochs is None or len(epochs) == 0:
        raise RuntimeError(
            "No valid epochs could be extracted from the input EEG file.\n"
            "The recording may be too short or all epochs were rejected.\n"
            "Check reject_peak_to_peak_uv and epoch_duration_sec in config."
        )

    feature_df = extract_features(epochs, cfg, subject_id="inference")
    return raw_clean, epochs, feature_df


def _predict_task_from_features(task: str, cfg: Config, feature_df: pd.DataFrame) -> dict[str, Any]:
    artifact = load_model_artifact(task, cfg)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names: list[str] = artifact["feature_names"]
    threshold = float(artifact["threshold"])

    aligned_df, alignment_mode = _prepare_inference_features(feature_df, feature_names)
    X = aligned_df[feature_names].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]

    mean_proba = float(np.mean(probas))
    predicted_label = int(mean_proba >= threshold)
    model_margin = float(mean_proba - threshold)

    return {
        "task": task,
        "predicted_label": predicted_label,
        "predicted_class": _label_name(task, predicted_label),
        "probability": mean_proba,
        "threshold": threshold,
        "epoch_probabilities": probas.tolist(),
        "inference_mode": alignment_mode,
        "model_margin": model_margin,
    }


def _compute_band_power_maps(epochs, cfg: Config) -> dict[str, list[float]]:
    from scipy.signal import welch

    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    sfreq = float(epochs.info["sfreq"])
    n_fft = int(cfg.get("features", "psd_n_fft", default=256) or min(256, data.shape[-1]))
    bands = cfg.get("prediction", "topomap_bands", default={}) or cfg.get("features", "bands", default={})

    ch_mean = np.mean(data, axis=0)
    maps: dict[str, list[float]] = {band: [] for band in bands}

    for ch_signal in ch_mean:
        freqs, psd = welch(ch_signal, fs=sfreq, nperseg=min(n_fft, len(ch_signal)))
        for band, (lo, hi) in bands.items():
            idx = (freqs >= float(lo)) & (freqs < float(hi))
            power = float(_TRAPZ(psd[idx], freqs[idx])) if np.any(idx) else 0.0
            maps[band].append(power)

    for band, vals in maps.items():
        arr = np.asarray(vals, dtype=float)
        scale = np.nanmax(np.abs(arr)) + 1e-12
        maps[band] = (arr / scale).tolist()

    return maps


def _compute_domain_evidence_maps(band_maps: dict[str, list[float]]) -> dict[str, list[float]]:
    delta = np.asarray(band_maps.get("delta", []), dtype=float)
    theta = np.asarray(band_maps.get("theta", []), dtype=float)
    alpha = np.asarray(band_maps.get("alpha", []), dtype=float)
    beta = np.asarray(band_maps.get("beta", []), dtype=float)

    n = max(len(delta), len(theta), len(alpha), len(beta))
    if n == 0:
        return {"alzheimer": [], "depression": []}

    def pad(arr):
        if len(arr) == n:
            return arr
        out = np.zeros(n, dtype=float)
        out[: len(arr)] = arr
        return out

    delta, theta, alpha, beta = map(pad, [delta, theta, alpha, beta])

    alz_map = (delta + theta) - (alpha + beta)
    dep_map = alpha - theta

    return {
        "alzheimer": _normalize_map(alz_map).tolist(),
        "depression": _normalize_map(dep_map).tolist(),
    }


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    scale = float(np.nanmax(np.abs(arr)) + 1e-12)
    return arr / scale


def _safe_mean(vals) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=float)))


def _build_topomap_analysis(
    channels: list[str],
    band_maps: dict[str, list[float]],
    domain_maps: dict[str, list[float]],
) -> dict[str, Any]:
    ch = [str(c).upper() for c in channels]
    region_idx = _region_indices(ch)

    band_means: dict[str, float] = {}
    regional_band_means: dict[str, dict[str, float]] = {}
    for band, vals in band_maps.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue
        band_means[band] = float(np.mean(arr))
        regional_band_means[band] = {
            region: _safe_region_mean(arr, idxs)
            for region, idxs in region_idx.items()
        }

    dominant_band = None
    if band_means:
        dominant_band = max(band_means, key=band_means.get)

    slowing_index = float(
        band_means.get("delta", 0.0)
        + band_means.get("theta", 0.0)
        - band_means.get("alpha", 0.0)
        - band_means.get("beta", 0.0)
    )

    alz_domain = float(np.mean(np.asarray(domain_maps.get("alzheimer", [0.0]), dtype=float)))
    dep_domain = float(np.mean(np.asarray(domain_maps.get("depression", [0.0]), dtype=float)))

    dep_arr = np.asarray(domain_maps.get("depression", []), dtype=float)
    alz_arr = np.asarray(domain_maps.get("alzheimer", []), dtype=float)
    frontal_dep = _safe_region_mean(dep_arr, region_idx.get("frontal", []))
    posterior_alz = _safe_region_mean(alz_arr, region_idx.get("posterior", []))
    temporal_dep = _safe_region_mean(dep_arr, region_idx.get("temporal", []))

    notes = []
    if dominant_band is not None:
        notes.append(
            f"Dominant normalized scalp band activity is {dominant_band} (mean={band_means[dominant_band]:.3f})."
        )

    notes.append(
        "Global slowing index (delta+theta-alpha-beta) is "
        f"{slowing_index:.3f}; higher values indicate relatively slower rhythms."
    )

    notes.append(
        "Alzheimer evidence map mean is "
        f"{alz_domain:.3f} with posterior mean {posterior_alz:.3f}."
    )

    notes.append(
        "Depression evidence map mean is "
        f"{dep_domain:.3f} with frontal mean {frontal_dep:.3f} and temporal mean {temporal_dep:.3f}."
    )

    return {
        "dominant_band": dominant_band,
        "global_slowing_index": slowing_index,
        "alzheimer_domain_mean": alz_domain,
        "depression_domain_mean": dep_domain,
        "posterior_alzheimer_mean": posterior_alz,
        "frontal_depression_mean": frontal_dep,
        "temporal_depression_mean": temporal_dep,
        "regional_band_means": regional_band_means,
        "clinical_notes": notes,
    }


def _build_medical_interpretation(
    analysis: dict[str, Any],
    decision: dict[str, str],
    alz_score: float,
    dep_score: float,
) -> dict[str, Any]:
    slowing = float(analysis.get("global_slowing_index", 0.0))
    dom_band = analysis.get("dominant_band") or "undetermined"
    alz_mean = float(analysis.get("alzheimer_domain_mean", 0.0))
    dep_mean = float(analysis.get("depression_domain_mean", 0.0))
    post_alz = float(analysis.get("posterior_alzheimer_mean", 0.0))
    front_dep = float(analysis.get("frontal_depression_mean", 0.0))
    temp_dep = float(analysis.get("temporal_depression_mean", 0.0))

    def _level(x: float) -> str:
        ax = abs(x)
        if ax < 0.05:
            return "mild"
        if ax < 0.15:
            return "moderate"
        return "marked"

    rhythm_text = (
        f"Dominant normalized rhythm is {dom_band}. "
        f"Global slowing index is {slowing:.3f} ({_level(slowing)} magnitude)."
    )

    alz_text = (
        f"Alzheimer-oriented map mean is {alz_mean:.3f}, with posterior emphasis {post_alz:.3f}. "
        "Higher posterior slowing-related values support Alzheimer-side evidence."
    )

    dep_text = (
        f"Depression-oriented map mean is {dep_mean:.3f}, with frontal {front_dep:.3f} "
        f"and temporal {temp_dep:.3f}. "
        "Frontal-temporal asymmetry-related predominance supports depression-side evidence."
    )

    gap = abs(alz_score - dep_score)
    stronger = "Alzheimer" if alz_score > dep_score else "Depression"
    integrated = (
        f"Integrated evidence score gap is {gap:.3f}; stronger side is {stronger}. "
        f"Final triage output: {decision.get('label', 'Inconclusive')} "
        f"({decision.get('reason', '').strip()})."
    )

    summary = (
        f"Topomap analysis indicates a {dom_band} dominant profile with "
        f"slowing index {slowing:.3f}. Combined model-evidence maps favor "
        f"{decision.get('label', 'Inconclusive')} for this EEG sample."
    )

    sections = [
        {"title": "Rhythm Profile", "text": rhythm_text},
        {"title": "Alzheimer Pattern", "text": alz_text},
        {"title": "Depression Pattern", "text": dep_text},
        {"title": "Integrated Impression", "text": integrated},
    ]

    return {
        "summary": summary,
        "sections": sections,
    }


def _build_clinical_recommendations(
    analysis: dict[str, Any],
    decision: dict[str, str],
    alz_score: float,
    dep_score: float,
) -> dict[str, Any]:
    label = decision.get("label", "Inconclusive")
    slowing = float(analysis.get("global_slowing_index", 0.0))
    post_alz = float(analysis.get("posterior_alzheimer_mean", 0.0))
    front_dep = float(analysis.get("frontal_depression_mean", 0.0))
    temp_dep = float(analysis.get("temporal_depression_mean", 0.0))
    gap = float(abs(alz_score - dep_score))

    urgency = "Routine follow-up"
    if gap >= 0.15:
        urgency = "Priority specialist review"
    elif label == "Inconclusive":
        urgency = "Early repeat assessment"

    common_steps = [
        "Correlate EEG findings with structured clinical interview and neurological/psychiatric examination.",
        "Review medication, sleep quality, substance use, and acute stressors that can alter EEG rhythms.",
        "Verify signal quality and montage coverage before final diagnostic decisions.",
    ]

    if label == "Alzheimer":
        targeted = [
            "Arrange cognitive screening and comprehensive neuropsychological assessment.",
            "Consider dementia workup with structural imaging and laboratory exclusion panel.",
            "Plan neurology or memory-clinic referral for confirmatory diagnostic pathway.",
        ]
        if slowing > 0.1 or post_alz > 0.1:
            targeted.append(
                "Posterior slowing pattern is relatively pronounced; prioritize evaluation for neurodegenerative etiology."
            )
    elif label == "Depression":
        targeted = [
            "Complete depression severity scoring and risk assessment (including suicidality as clinically indicated).",
            "Coordinate psychiatry consultation for treatment planning (psychotherapy/pharmacotherapy options).",
            "Schedule short-interval follow-up to track symptom trajectory and treatment response.",
        ]
        if front_dep > 0.1 or temp_dep > 0.1:
            targeted.append(
                "Frontal-temporal evidence is relatively elevated; monitor affective and executive symptom domains closely."
            )
    else:
        targeted = [
            "Do not anchor diagnosis on current EEG alone; evidence is not clearly separable between disease models.",
            "Repeat EEG under controlled conditions and compare with longitudinal clinical findings.",
            "Use multidisciplinary review (neurology + psychiatry) before assigning disease-specific label.",
        ]

    return {
        "urgency": urgency,
        "evidence_gap": gap,
        "recommended_next_steps": common_steps + targeted,
    }


def _region_indices(channels_upper: list[str]) -> dict[str, list[int]]:
    idx = {
        "frontal": [],
        "temporal": [],
        "central": [],
        "parietal": [],
        "occipital": [],
        "posterior": [],
    }
    for i, ch in enumerate(channels_upper):
        if ch.startswith(("FP", "AF", "F")):
            idx["frontal"].append(i)
        if ch.startswith("T"):
            idx["temporal"].append(i)
        if ch.startswith("C"):
            idx["central"].append(i)
        if ch.startswith("P"):
            idx["parietal"].append(i)
            idx["posterior"].append(i)
        if ch.startswith("O"):
            idx["occipital"].append(i)
            idx["posterior"].append(i)
    return idx


def _safe_region_mean(arr: np.ndarray, indices: list[int]) -> float:
    if arr.size == 0 or not indices:
        return 0.0
    use = [i for i in indices if i < arr.size]
    if not use:
        return 0.0
    return float(np.mean(arr[use]))


def _prepare_inference_features(
    feature_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, str]:
    """
    Prepare inference features to match training schema.

    Returns
    -------
    (aligned_df, mode)
        mode is "epoch_level" when direct overlap is strong, else
        "subject_aggregated" when we reconstruct subject-level aggregate features.
    """
    raw_feature_df = feature_df.drop(columns=["subject_id"], errors="ignore")
    overlap = len([c for c in feature_names if c in raw_feature_df.columns])
    overlap_ratio = overlap / max(1, len(feature_names))

    if overlap_ratio >= 0.5:
        return _align_features(raw_feature_df, feature_names), "epoch_level"

    aggregated = _aggregate_to_training_schema(raw_feature_df, feature_names)
    aligned = _align_features(aggregated, feature_names)
    return aligned, "subject_aggregated"


def _aggregate_to_training_schema(
    feature_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Rebuild one-row subject-level features from per-epoch rows when
    training used aggregated schema names like <feature>_<agg>.
    """
    agg_funcs = {
        "mean": "mean",
        "std": "std",
        "median": "median",
        "min": "min",
        "max": "max",
        "var": "var",
    }

    row: dict[str, float] = {}
    for feat in feature_names:
        if feat in feature_df.columns:
            # If exact column exists, aggregate with mean as a safe fallback.
            row[feat] = float(pd.to_numeric(feature_df[feat], errors="coerce").mean())
            continue

        base, sep, suffix = feat.rpartition("_")
        if not sep or suffix not in agg_funcs or base not in feature_df.columns:
            continue

        series = pd.to_numeric(feature_df[base], errors="coerce")
        val = series.agg(agg_funcs[suffix])
        if pd.isna(val):
            continue
        row[feat] = float(val)

    if not row:
        logger.warning(
            "Could not reconstruct aggregated inference schema; falling back to raw alignment."
        )
        return feature_df

    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────────────
def _align_features(
    feature_df: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    """
    Align extracted features to the expected training schema.

    * Missing columns are filled with 0.0.
    * Extra columns are dropped.
    * Column order is fixed to match training.
    """
    missing = [c for c in feature_names if c not in feature_df.columns]
    if missing:
        logger.warning(
            "Inference feature alignment: %d columns missing from extracted "
            "features (filling with 0): %s…",
            len(missing),
            missing[:5],
        )
        for col in missing:
            feature_df[col] = 0.0

    return feature_df[feature_names]


def _label_name(task: str, label: int) -> str:
    """Map numeric label to human-readable class name."""
    names = {
        "alzheimer": {1: "Alzheimer's Positive", 0: "Healthy Control"},
        "depression": {1: "Depression (MDD)", 0: "Non-depressed"},
    }
    return names.get(task, {}).get(label, str(label))
