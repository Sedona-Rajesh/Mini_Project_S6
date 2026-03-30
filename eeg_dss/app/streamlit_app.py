"""Streamlit UI for dual-disease EEG triage."""

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
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(0, 255, 214, 0.08), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(0, 173, 255, 0.10), transparent 35%),
                linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
            color: #dbe8ff;
        }
        .block-container { padding-top: 1.2rem; }
        .clinic-card {
            background: rgba(16, 24, 48, 0.75);
            border: 1px solid rgba(0, 255, 214, 0.35);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 0 0 1px rgba(0, 255, 214, 0.12), 0 0 24px rgba(0, 173, 255, 0.18);
        }
        .final-chip {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            border: 1px solid rgba(57, 255, 20, 0.8);
            background: rgba(57, 255, 20, 0.12);
            color: #b7ff9b;
            box-shadow: 0 0 14px rgba(57, 255, 20, 0.35);
        }
        .neon-box {
            background: rgba(8, 14, 30, 0.8);
            border-left: 4px solid #00ffd6;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin: 0.35rem 0 0.6rem 0;
            color: #d5e8ff;
        }
        .neon-box b {
            color: #8ce9ff;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #00ffd6, #00adff) !important;
            color: #04111f !important;
            border: 1px solid #00ffd6 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            box-shadow: 0 0 18px rgba(0, 255, 214, 0.35) !important;
        }
        .stMetric {
            background: rgba(14, 22, 44, 0.7);
            border: 1px solid rgba(0, 173, 255, 0.35);
            border-radius: 12px;
            padding: 0.6rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1227 0%, #0b1735 100%);
            border-right: 1px solid rgba(0, 173, 255, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_text_box(text: str, title: str | None = None) -> None:
    if not text:
        return
    prefix = f"<b>{title}</b><br>" if title else ""
    st.markdown(f"<div class='neon-box'>{prefix}{text}</div>", unsafe_allow_html=True)


cfg = _load_config(CONFIG_PATH)
_inject_ui_styles()

try:
    alz_artifact = _load_artifact("alzheimer", CONFIG_PATH)
    dep_artifact = _load_artifact("depression", CONFIG_PATH)
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Model loading failed: {e}")

st.title("EEG Dual-Disease Triage")
st.caption("Single upload workflow for Alzheimer vs Depression decision support")

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

    tab_summary, tab_details, tab_visuals = st.tabs([
        "🩺 Clinical Summary", 
        "📊 Analysis Details", 
        "🗺️ Visualizations"
    ])

    with tab_summary:
        st.markdown(
            f"""
            <div class='clinic-card'>
                <span class='final-chip'>Final Prediction: {result['final_prediction']}</span>
                <p style='margin-top: 1rem; margin-bottom: 0;'><strong>Decision Rationale:</strong> {result['decision_reason']}</p>
            </div><br>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Medical Interpretation")
        interp = result.get("medical_interpretation", {})
        summary = interp.get("summary")
        
        mi_html = "<div class='clinic-card'>"
        if summary:
            mi_html += f"<p style='margin-bottom: 0.5rem;'><b>Summary:</b> {summary}</p><hr style='margin: 0.5rem 0; border-color: rgba(255,255,255,0.1);'/>"

        sections = interp.get("sections", [])
        if sections:
            for section in sections:
                title = section.get("title", "")
                text = section.get("text", "")
                if title and text:
                    mi_html += f"<p style='margin-bottom: 0.5rem;'><b>{title}:</b> {text}</p>"
        mi_html += "</div><br>"
        st.markdown(mi_html, unsafe_allow_html=True)

        st.subheader("Clinical Next Steps")
        rec = result.get("clinical_recommendations", {})
        urgency = rec.get("urgency")
        if urgency:
            if "Routine" in urgency:
                st.info(f"**Recommended Priority:** {urgency}")
            else:
                st.warning(f"**Recommended Priority:** {urgency}")

        next_steps = rec.get("recommended_next_steps", [])
        if next_steps:
            ns_html = "<div class='clinic-card'><ul style='margin-bottom: 0;'>"
            for step in next_steps:
                ns_html += f"<li>{step}</li>"
            ns_html += "</ul></div><br>"
            st.markdown(ns_html, unsafe_allow_html=True)

    with tab_details:
        st.subheader("Model Probabilities & Scores")
        col_a, col_d = st.columns(2)
        with col_a:
            st.subheader("Alzheimer Model")
            st.metric("Probability", f"{result['alzheimer']['probability']:.1%}")
            st.metric("Evidence Score", f"{result['alzheimer']['evidence_score']:.3f}")
            st.caption(result["alzheimer"]["predicted_class"])
        with col_d:
            st.subheader("Depression Model")
            st.metric("Probability", f"{result['depression']['probability']:.1%}")
            st.metric("Evidence Score", f"{result['depression']['evidence_score']:.3f}")
            st.caption(result["depression"]["predicted_class"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Topomap Interpretation")
        analysis = result.get("scalp", {}).get("analysis", {})
        notes = analysis.get("clinical_notes", [])
        if notes:
            for note in notes:
                _render_text_box(note, title="Interpretation")

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

    with tab_visuals:
        with st.expander("Scalp Electrode Layout", expanded=False):
            _render_text_box(
                "Electrode layout map is positional only. It confirms channel coverage and scalp locations, not pathology intensity.",
                title="How To Read This Panel",
            )
            fig_elec = plot_electrode_positions(result["scalp"]["channels"])
            st.pyplot(fig_elec, use_container_width=True)

        with st.expander("Band-Power Topomaps", expanded=True):
            _render_text_box(
                "Color meaning (band power maps): darker blue/purple indicates lower relative band power, "
                "while green/yellow indicates higher relative band power for that band within this EEG sample.",
                title="Color Legend",
            )
            band_cols = st.columns(2)
            band_names = list(result["scalp"]["band_power_maps"].keys())
            for i, band in enumerate(band_names[:4]):
                with band_cols[i % 2]:
                    fig_band = plot_scalp_topomap(
                        result["scalp"]["channels"],
                        result["scalp"]["band_power_maps"][band],
                        title=f"{band.capitalize()} Power",
                        cmap="viridis",
                        colorbar_label="Relative band power (normalized)",
                    )
                    st.pyplot(fig_band, use_container_width=True)

        with st.expander("Model Evidence Topomaps", expanded=True):
            _render_text_box(
                "Color meaning (evidence maps): red indicates stronger model-supporting evidence for the shown condition, "
                "blue indicates weaker or opposite evidence, and near-white indicates neutral/low contribution.",
                title="Color Legend",
            )
            e1, e2 = st.columns(2)
            with e1:
                fig_alz_map = plot_scalp_topomap(
                    result["scalp"]["channels"],
                    result["scalp"]["domain_evidence_maps"]["alzheimer"],
                    title="Alzheimer Evidence Map",
                    cmap="RdBu_r",
                    colorbar_label="Alzheimer evidence (normalized)",
                )
                st.pyplot(fig_alz_map, use_container_width=True)
            with e2:
                fig_dep_map = plot_scalp_topomap(
                    result["scalp"]["channels"],
                    result["scalp"]["domain_evidence_maps"]["depression"],
                    title="Depression Evidence Map",
                    cmap="RdBu_r",
                    colorbar_label="Depression evidence (normalized)",
                )
                st.pyplot(fig_dep_map, use_container_width=True)

        with st.expander("PSD Preview", expanded=False):
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
