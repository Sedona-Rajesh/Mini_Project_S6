"""Streamlit UI for authenticated, dual-disease EEG triage."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config must come first ───────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Decision Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Resolve config path from CLI args ────────────────────────────────────────
def _parse_config_arg() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/config.yaml")
    args, _ = parser.parse_known_args()
    return args.config


CONFIG_PATH = _parse_config_arg()

# ── Lazy imports (avoid heavy imports at module level for faster cold starts) ─
@st.cache_resource(show_spinner=False)
def _load_config(config_path: str):
    from eeg_dss.config import load_config
    return load_config(config_path)


@st.cache_resource(show_spinner=False)
def _load_artifact(task: str, config_path: str):
    from eeg_dss.config import load_config
    from eeg_dss.training.trainer import load_model_artifact
    cfg = load_config(config_path)
    return load_model_artifact(task, cfg)


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f7fbfc 0%, #eef5f8 100%); }
        .block-container { padding-top: 1.2rem; }
        .clinic-card {
            background: #ffffff;
            border: 1px solid #d8e7ec;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(16, 77, 96, 0.08);
        }
        .final-chip {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            border: 1px solid #b9d7de;
            background: #e9f6f9;
            color: #14526b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


cfg = _load_config(CONFIG_PATH)
_inject_ui_styles()

with st.sidebar:
    st.title("NeuroClinic DSS")
    st.caption("EEG Triage Workstation")
    st.divider()
    st.code(CONFIG_PATH, language=None)

try:
    alz_artifact = _load_artifact("alzheimer", CONFIG_PATH)
    dep_artifact = _load_artifact("depression", CONFIG_PATH)
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Model loading failed: {e}")

st.title("EEG Dual-Disease Triage")
st.caption("Single upload workflow for Alzheimer vs Depression decision support")

if model_ready:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.metric("Alzheimer Features", len(alz_artifact["feature_names"]))
        alz_model = (
            alz_artifact.get("metadata", {}).get("selected_model")
            or "unknown"
        )
        st.caption(f"Selected model: {alz_model}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.metric("Depression Features", len(dep_artifact["feature_names"]))
        dep_model = (
            dep_artifact.get("metadata", {}).get("selected_model")
            or "unknown"
        )
        st.caption(f"Selected model: {dep_model}")
        st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Upload EEG")
uploaded_files = st.file_uploader(
    "Upload one EEG file; for .set upload matching .fdt too.",
    type=["set", "fdt", "edf", "bdf", "fif", "vhdr"],
    accept_multiple_files=True,
    disabled=not model_ready,
)

if uploaded_files and model_ready:
    eeg_files = [f for f in uploaded_files if Path(f.name).suffix.lower() != ".fdt"]
    if len(eeg_files) != 1:
        st.error("Please upload exactly one EEG file (.set/.edf/.bdf/.fif/.vhdr).")
        st.stop()

    uploaded_file = eeg_files[0]
    ext = Path(uploaded_file.name).suffix.lower()
    temp_dir = Path(tempfile.mkdtemp(prefix="eeg_dss_"))
    input_path = temp_dir / uploaded_file.name
    input_path.write_bytes(uploaded_file.getbuffer())

    for sidecar in uploaded_files:
        if Path(sidecar.name).suffix.lower() == ".fdt":
            (temp_dir / sidecar.name).write_bytes(sidecar.getbuffer())

    from eeg_dss.prediction.predictor import predict_dual_from_file
    from eeg_dss.visualization import (
        plot_electrode_positions,
        plot_raw_psd,
        plot_scalp_topomap,
    )
    from eeg_dss.data.loader import load_raw

    with st.spinner("Running dual-model triage..."):
        try:
            result = predict_dual_from_file(input_path, cfg)
            inference_error = None
        except Exception as exc:
            result = None
            inference_error = str(exc)

    if inference_error:
        err_text = str(inference_error)
        if ext == ".set" and ".fdt" in err_text.lower():
            st.error(
                "Inference failed because this EEGLAB `.set` references an external `.fdt` file. "
                "If available, upload the matching `.fdt` sidecar along with the `.set` file."
            )
            st.caption(err_text)
        else:
            st.error(f"Inference failed: {err_text}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.stop()

    st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='final-chip'>Final Prediction: {result['final_prediction']}</span>", unsafe_allow_html=True)
    st.write(result["decision_reason"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Medical Interpretation")
    interp = result.get("medical_interpretation", {})
    summary = interp.get("summary")
    if summary:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.write(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    for section in interp.get("sections", []):
        title = section.get("title")
        text = section.get("text")
        if title and text:
            st.markdown(f"**{title}**")
            st.write(text)

    st.subheader("Clinical Next Steps")
    rec = result.get("clinical_recommendations", {})
    urgency = rec.get("urgency")
    if urgency:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.write(f"**Recommended Priority:** {urgency}")
        st.markdown("</div>", unsafe_allow_html=True)

    next_steps = rec.get("recommended_next_steps", [])
    if next_steps:
        for step in next_steps:
            st.markdown(f"- {step}")

    col_a, col_d = st.columns(2)
    with col_a:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.subheader("Alzheimer Model")
        st.metric("Probability", f"{result['alzheimer']['probability']:.1%}")
        st.metric("Evidence Score", f"{result['alzheimer']['evidence_score']:.3f}")
        st.caption(result["alzheimer"]["predicted_class"])
        st.markdown("</div>", unsafe_allow_html=True)
    with col_d:
        st.markdown("<div class='clinic-card'>", unsafe_allow_html=True)
        st.subheader("Depression Model")
        st.metric("Probability", f"{result['depression']['probability']:.1%}")
        st.metric("Evidence Score", f"{result['depression']['evidence_score']:.3f}")
        st.caption(result["depression"]["predicted_class"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Scalp Electrode Layout")
    fig_elec = plot_electrode_positions(result["scalp"]["channels"])
    st.pyplot(fig_elec, use_container_width=True)

    st.subheader("Band-Power Topomaps")
    band_cols = st.columns(2)
    band_names = list(result["scalp"]["band_power_maps"].keys())
    for i, band in enumerate(band_names[:4]):
        with band_cols[i % 2]:
            fig_band = plot_scalp_topomap(
                result["scalp"]["channels"],
                result["scalp"]["band_power_maps"][band],
                title=f"{band.capitalize()} Power",
                cmap="viridis",
            )
            st.pyplot(fig_band, use_container_width=True)

    st.subheader("Model Evidence Topomaps")
    e1, e2 = st.columns(2)
    with e1:
        fig_alz_map = plot_scalp_topomap(
            result["scalp"]["channels"],
            result["scalp"]["domain_evidence_maps"]["alzheimer"],
            title="Alzheimer Evidence Map",
            cmap="RdBu_r",
        )
        st.pyplot(fig_alz_map, use_container_width=True)
    with e2:
        fig_dep_map = plot_scalp_topomap(
            result["scalp"]["channels"],
            result["scalp"]["domain_evidence_maps"]["depression"],
            title="Depression Evidence Map",
            cmap="RdBu_r",
        )
        st.pyplot(fig_dep_map, use_container_width=True)

    st.subheader("Topomap Interpretation")
    analysis = result.get("scalp", {}).get("analysis", {})
    notes = analysis.get("clinical_notes", [])
    if notes:
        for note in notes:
            st.markdown(f"- {note}")

    band_region = analysis.get("regional_band_means", {})
    if band_region:
        rows = []
        for band, regions in band_region.items():
            row = {"band": band}
            row.update(regions)
            rows.append(row)
        if rows:
            import pandas as pd

            st.caption("Regional normalized mean values extracted from current scalp maps")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("PSD Preview")
    raw_preview = load_raw(input_path)
    fig_psd = plot_raw_psd(raw_preview, task="triage")
    st.pyplot(fig_psd, use_container_width=True)

    st.download_button(
        label="Download Triage JSON",
        data=json.dumps(result, indent=2),
        file_name=f"triage_result_{uploaded_file.name}.json",
        mime="application/json",
    )

    shutil.rmtree(temp_dir, ignore_errors=True)
elif not model_ready:
    st.info("Train/load both task models to enable dual triage.")
else:
    st.info("Upload EEG data to start analysis.")
