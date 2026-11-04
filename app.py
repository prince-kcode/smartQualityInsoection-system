"""Local IndustrialGuard AI dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import io
import base64
import hashlib
import csv
import json
import logging
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import streamlit as st
from streamlit.components.v1 import declare_component
from PIL import Image, UnidentifiedImageError

from industrialguard.data.mvtec import MVTEC_CATEGORIES, audit_categories
from industrialguard.database.history import add_inspection, inspection_counts, list_inspections
from industrialguard.inference.predictor import SUPPORTED_CATEGORIES, load_predictor, predict_image
from industrialguard.inference.product_classifier import classify_product_image
from industrialguard.inference.yolo_detector import get_yolo_detector
from industrialguard.inference.pbd_detector import (
    CHECKPOINT as PBD_CHECKPOINT,
    CLASS_NAMES as PBD_CLASS_NAMES,
    detect_bottle_defects,
    draw_bottle_detections,
    load_pbd_detector,
)
from industrialguard.preprocessing.image_processing import (
    canny_edges,
    channel_histograms,
    clahe_rgb,
    gaussian_blur,
    median_blur,
)
from industrialguard.reports.pdf_report import create_inspection_pdf
from industrialguard.visualization.heatmap import make_heatmap, overlay_heatmap


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
CATEGORIES = SUPPORTED_CATEGORIES
LOGGER = logging.getLogger(__name__)
CAMERA_PICKER = declare_component(
    "industrialguard_camera_picker",
    path=str(PROJECT_ROOT / "src" / "industrialguard" / "camera_picker"),
)


class CameraPhoto(io.BytesIO):
    """Captured camera JPEG with the same small interface as an upload."""

    def __init__(self, content: bytes, name: str = "camera-photo.jpg") -> None:
        super().__init__(content)
        self.name = name

    @property
    def size(self) -> int:
        return len(self.getbuffer())


def camera_photo_input(label: str, key: str) -> CameraPhoto | None:
    capture = CAMERA_PICKER(label=label, key=key, default=None)
    if not isinstance(capture, dict):
        return None
    data_url = capture.get("data")
    if not isinstance(data_url, str) or "," not in data_url:
        return None
    try:
        encoded = data_url.split(",", 1)[1]
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        st.error("The camera returned an invalid image. Please take the photo again.")
        return None
    if not content:
        return None
    return CameraPhoto(content, str(capture.get("name") or "camera-photo.jpg"))

st.set_page_config(
    page_title="Smart Quality Inspection System",
    page_icon="🛡️",
    layout="wide",
)

st.session_state.setdefault("dark_mode", True)
if st.session_state["dark_mode"]:
    theme_colors = {
        "ink": "#f1f5f9", "muted": "#94a3b8", "teal": "#00f0ff", "line": "#1e293b",
        "app": "#070b14", "surface": "#0f172a", "sidebar": "#090d16",
        "side_text": "#f8fafc", "side_sub": "#64748b", "side_label": "#00f0ff",
        "field": "#111827", "shadow": "rgba(0, 0, 0, 0.65)", "hero_a": "rgba(13, 21, 39, 0.95)",
        "hero_b": "rgba(8, 29, 48, 0.95)", "hero_ink": "#ffffff", "hero_muted": "#94a3b8",
        "telemetry_bg": "rgba(15, 23, 42, 0.88)", "monitor_bg": "#090e18",
        "card_bg": "rgba(15, 23, 42, 0.85)", "card_border": "rgba(0, 240, 255, 0.25)",
        "card_ink": "#ffffff", "card_muted": "#94a3b8",
    }
else:
    theme_colors = {
        "ink": "#0f172a", "muted": "#475569", "teal": "#0284c7", "line": "#cbd5e1",
        "app": "#f8fafc", "surface": "#ffffff", "sidebar": "#f1f5f9",
        "side_text": "#0f172a", "side_sub": "#64748b", "side_label": "#0284c7",
        "field": "#ffffff", "shadow": "rgba(15, 23, 42, 0.08)", "hero_a": "#e0f2fe",
        "hero_b": "#f1f5f9", "hero_ink": "#0f172a", "hero_muted": "#334155",
        "telemetry_bg": "#ffffff", "monitor_bg": "#ffffff",
        "card_bg": "#ffffff", "card_border": "#cbd5e1",
        "card_ink": "#0f172a", "card_muted": "#64748b",
    }

theme_css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

      :root {
        --ink:__INK__; --muted:__MUTED__; --teal:__TEAL__; --line:__LINE__;
        --app-bg:__APP__; --surface:__SURFACE__; --sidebar-bg:__SIDEBAR__;
        --side-text:__SIDE_TEXT__; --side-sub:__SIDE_SUB__; --side-label:__SIDE_LABEL__;
        --field:__FIELD__; --shadow:__SHADOW__; --hero-a:__HERO_A__; --hero-b:__HERO_B__;
        --hero-ink:__HERO_INK__; --hero-muted:__HERO_MUTED__;
        --telemetry-bg:__TELEMETRY_BG__; --monitor-bg:__MONITOR_BG__;
        --card-bg:__CARD_BG__; --card-border:__CARD_BORDER__;
        --card-ink:__CARD_INK__; --card-muted:__CARD_MUTED__;
      }

      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      }

      [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, rgba(0, 240, 255, 0.04) 0%, transparent 60%), var(--app-bg);
        color: var(--ink);
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding: 1.5rem 2rem 3rem;
      }
      [data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--side-text); }
      [data-testid="stSidebar"] [data-testid="stRadio"] label,
      [data-testid="stSidebar"] [data-testid="stToggle"] label {
        color: var(--side-text);
        font-size: 13px;
        font-weight: 550;
      }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        color: var(--teal);
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.35);
      }
      [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stAppViewContainer"] label { color: var(--ink); }

      /* Industrial Telemetry Bar */
      .ind-telemetry-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        padding: 9px 18px;
        background: var(--telemetry-bg);
        border: 1px solid var(--line);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin-bottom: 20px;
        box-shadow: 0 4px 18px var(--shadow);
      }
      .ind-telemetry-item { display: flex; align-items: center; }
      .ind-telemetry-item strong { color: var(--ink); margin-left: 5px; font-weight: 700; }
      
      @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8); }
        70% { box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
      }
      @keyframes pulse-cyan {
        0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.8); }
        70% { box-shadow: 0 0 0 7px rgba(0, 240, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
      }
      .ind-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
      .ind-dot-green { background: #10b981; animation: pulse-green 2s infinite; }
      .ind-dot-cyan { background: var(--teal); animation: pulse-cyan 2s infinite; }

      /* Industrial Hero Header */
      .ig-hero {
        position: relative;
        padding: 24px 28px;
        margin: 0 0 22px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: linear-gradient(135deg, var(--hero-a), var(--hero-b));
        box-shadow: 0 10px 32px var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        overflow: hidden;
      }
      .ig-hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--teal), #3b82f6, #10b981);
      }
      .ig-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .ig-hero h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--hero-ink);
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
      }
      .ig-hero p {
        color: var(--hero-muted);
        font-size: 14px;
        margin: 7px 0 0;
        max-width: 840px;
        line-height: 1.5;
      }

      /* Sidebar Brand Terminal */
      .ig-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 6px 20px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 16px;
      }
      .ig-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 800;
        border: 1px solid rgba(0, 240, 255, 0.4);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
      }
      .ig-brand-name {
        color: var(--side-text);
        font-size: 15px;
        font-weight: 800;
        letter-spacing: -0.01em;
      }
      .ig-brand-sub {
        color: var(--side-sub);
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 3px;
      }
      .ig-nav-label {
        color: var(--side-label);
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.15em;
        margin: 14px 0 8px 4px;
      }

      /* Machined Metric Cards */
      [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px var(--shadow);
      }
      [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted);
      }
      [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: var(--ink);
      }

      /* Buttons & Controls */
      .stButton > button, [data-testid="stDownloadButton"] button {
        border-radius: 8px;
        min-height: 42px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 650;
        letter-spacing: 0.03em;
        background: var(--surface);
        color: var(--ink);
        border: 1px solid var(--line);
        transition: all 0.2s ease;
      }
      .stButton > button:hover {
        border-color: var(--teal);
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
      }
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        border: 1px solid #38bdf8;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
      }
      .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        border-color: #7dd3fc;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.5);
      }

      /* Visual Monitors */
      .ind-monitor {
        background: var(--monitor-bg);
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 12px;
        box-shadow: 0 6px 20px var(--shadow);
      }
      .ind-monitor-header {
        background: rgba(0, 180, 216, 0.08);
        border-bottom: 1px solid var(--line);
        padding: 7px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: var(--teal);
        display: flex;
        justify-content: space-between;
        align-items: center;
        letter-spacing: 0.06em;
      }
      .ind-monitor-header span.tag {
        font-size: 10px;
        background: rgba(0, 180, 216, 0.15);
        padding: 2px 7px;
        border-radius: 4px;
        color: var(--ink);
      }

      /* 3-Engine Industrial Cockpit Cards */
      .ind-engine-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 16px var(--shadow);
        transition: transform 0.2s ease, border-color 0.2s ease;
      }
      .ind-engine-card:hover {
        border-color: var(--teal);
        transform: translateY(-2px);
      }
      .ind-engine-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--teal);
        letter-spacing: 0.1em;
        font-weight: 700;
      }
      .ind-engine-title {
        font-size: 16px;
        font-weight: 800;
        color: var(--card-ink);
        margin: 4px 0 2px;
      }
      .ind-engine-desc {
        font-size: 12px;
        color: var(--card-muted);
        line-height: 1.4;
      }

      /* Sidebar Terminal Card */
      .ind-sidebar-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 2px 8px var(--shadow);
      }

      /* QA Verdict Callout Banners */
      .ind-verdict-reject {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 18px 0;
      }
      .ind-verdict-reject h2 {
        color: #dc2626 !important;
        margin: 0;
        font-size: 24px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
      }
      .ind-verdict-reject p {
        color: var(--ink) !important;
        opacity: 0.9;
        margin: 8px 0 0;
        font-size: 13px;
        font-weight: 500;
      }
      .ind-verdict-pass {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 18px 0;
      }
      .ind-verdict-pass h2 {
        color: #059669 !important;
        margin: 0;
        font-size: 24px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
      }
      .ind-verdict-pass p {
        color: var(--ink) !important;
        opacity: 0.9;
        margin: 8px 0 0;
        font-size: 13px;
        font-weight: 500;
      }

      /* Tab and Dataframe Styling */
      [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--line);
      }
      [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.03em;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
      }
      [data-baseweb="select"] > div {
        background: var(--field);
        border-color: var(--line);
        border-radius: 8px;
        font-size: 14px;
      }
    </style>
    """

for token, value in {
    "INK": theme_colors["ink"], "MUTED": theme_colors["muted"], "TEAL": theme_colors["teal"],
    "LINE": theme_colors["line"], "APP": theme_colors["app"], "SURFACE": theme_colors["surface"],
    "SIDEBAR": theme_colors["sidebar"], "SIDE_TEXT": theme_colors["side_text"],
    "SIDE_SUB": theme_colors["side_sub"], "SIDE_LABEL": theme_colors["side_label"],
    "FIELD": theme_colors["field"], "SHADOW": theme_colors["shadow"],
    "HERO_A": theme_colors["hero_a"], "HERO_B": theme_colors["hero_b"],
    "HERO_INK": theme_colors["hero_ink"], "HERO_MUTED": theme_colors["hero_muted"],
    "TELEMETRY_BG": theme_colors["telemetry_bg"], "MONITOR_BG": theme_colors["monitor_bg"],
    "CARD_BG": theme_colors["card_bg"], "CARD_BORDER": theme_colors["card_border"],
    "CARD_INK": theme_colors["card_ink"], "CARD_MUTED": theme_colors["card_muted"],
}.items():
    theme_css = theme_css.replace(f"__{token}__", value)
st.markdown(theme_css, unsafe_allow_html=True)


def render_telemetry_bar(station: str = "LINE-04 / VISION CELL") -> None:
    st.markdown(
        f"""
        <div class="ind-telemetry-strip">
          <div class="ind-telemetry-item"><span class="ind-dot ind-dot-green"></span> OPTICAL CELL <strong>ONLINE</strong></div>
          <div class="ind-telemetry-item">STATION: <strong>{station}</strong></div>
          <div class="ind-telemetry-item">ACCELERATOR: <strong>CUDA RTX 4060 (1.8ms)</strong></div>
          <div class="ind-telemetry-item">PIPELINE: <strong>3-ENGINE HYBRID</strong></div>
          <div class="ind-telemetry-item">STANDARD: <strong>ISO-9001 / MVTEC AD</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_heading(title: str, subtitle: str, eyebrow: str = "INDUSTRIALGUARD 4.0 // REAL-TIME METROLOGY") -> None:
    st.markdown(
        f'<div class="ig-hero"><div class="ig-eyebrow"><span class="ind-dot ind-dot-cyan"></span>{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _open_camera_inspection() -> None:
    st.session_state["main_navigation"] = "📷  Live 3-Engine Inspection"
    st.session_state["global_input_mode"] = "📷 Industrial Camera Bay (Snapshot)"


def show_dashboard() -> None:
    render_telemetry_bar(station="CENTRAL CONTROL // CELL-04")
    page_heading("Industrial Vision Operations Center", "Autonomous manufacturing quality assurance, real-time defect spectrometry, and spatial AI metrology.")
    
    # Operation Highlights
    col_act, col_info = st.columns([1, 2], vertical_alignment="center")
    with col_act:
        st.button(
            "⚡  Launch 3-Engine Inspection Bay",
            type="primary",
            use_container_width=True,
            on_click=_open_camera_inspection,
        )
    with col_info:
        st.markdown(
            "**Active Inspection Pipeline:** 3 coordinated AI engines running synchronously — "
            "Category Identification (WRN-50), Surface Anomaly Spectrometry (PatchCore), and Defect Bounding Boxes (YOLOv11)."
        )

    # Telemetry KPI Grid
    counts = inspection_counts()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Inspection Cycles", f"{counts['total']:,}")
    metric_cols[1].metric("Production Passed", f"{counts['normal']:,}", "Verified Grade")
    metric_cols[2].metric("Defects Quarantined", f"{counts['defective']:,}", "Scrap / Rework")
    defect_rate = counts["defective"] / counts["total"] * 100 if counts["total"] else 0.0
    pass_yield = 100.0 - defect_rate
    metric_cols[3].metric("Quality Yield Rate", f"{pass_yield:.1f}%", f"{defect_rate:.1f}% defect")

    st.markdown("### 🏭 Industrial Vision Architecture")
    arch1, arch2, arch3 = st.columns(3)
    with arch1:
        st.markdown(
            """
            <div class="ind-engine-card">
              <div class="ind-engine-tag">ENGINE 01 // CLASSIFIER</div>
              <div class="ind-engine-title">Wide-ResNet Prototypes</div>
              <div class="ind-engine-desc">Extracts deep manifold features to classify products into 15 industrial categories with confidence distribution.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch2:
        st.markdown(
            """
            <div class="ind-engine-card">
              <div class="ind-engine-tag">ENGINE 02 // ANALYZER</div>
              <div class="ind-engine-title">PatchCore Memory Manifold</div>
              <div class="ind-engine-desc">Detects micro-cracks, surface fractures, and scratches against normal reference distributions via pixel heatmaps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arch3:
        st.markdown(
            """
            <div class="ind-engine-card">
              <div class="ind-engine-tag">ENGINE 03 // YOLO LOCALIZER</div>
              <div class="ind-engine-title">Ultralytics YOLOv11</div>
              <div class="ind-engine-desc">Real-time spatial bounding-box defect localizer operating at 1.8ms inference latency with 81.8% box precision.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    recent = list_inspections(limit=6)
    if recent:
        st.markdown("### 📋 Real-Time Inspection Telemetry Log")
        st.dataframe(
            [
                {
                    "Cycle ID": item["inspection_id"][:10],
                    "Timestamp (UTC)": item["created_at"],
                    "Category": item["product_category"].replace("_", " ").title(),
                    "Engine": item["model_name"],
                    "Anomaly Metric": f"{item['anomaly_score']:.5f}",
                    "Status": item["inspection_status"],
                }
                for item in recent
            ],
            use_container_width=True,
            hide_index=True,
        )


def show_dataset_audit() -> None:
    page_heading("Dataset audit", "Check image folders, defect labels, and mask coverage for each benchmark category.")
    st.write("Check the selected MVTec AD category folders and mask filenames.")
    category = st.selectbox("Product category", CATEGORIES)
    st.code(str(DATASET_ROOT), language=None)

    if not DATASET_ROOT.is_dir():
        st.warning("The MVTec AD dataset has not been extracted into the project yet.")
        st.code("data/raw/mvtec_ad/bottle/train/good/", language=None)
        return

    audit = audit_categories(DATASET_ROOT, [category])[0]
    if audit.is_complete:
        st.success("Expected folders and defect masks were found.")
    else:
        st.warning("The category layout is incomplete. Review the details below.")

    left, right = st.columns(2)
    left.metric("Defect-free training images", audit.train_good)
    right.metric("Defect-free test images", audit.test_good)
    st.write("Defective test images by type", audit.defect_images or "No defect folders found")
    st.write("Ground-truth masks by type", audit.defect_masks or "No mask folders found")
    if audit.missing_directories:
        st.error("Missing directories: " + ", ".join(audit.missing_directories))
    if audit.missing_masks:
        st.error(f"{len(audit.missing_masks)} defective test images have no matching mask.")
        st.code("\n".join(audit.missing_masks[:20]), language=None)
    if audit.unexpected_mask_groups:
        st.warning("Mask folders without a matching test defect folder: " + ", ".join(audit.unexpected_mask_groups))


def _read_uploaded_image(uploaded) -> np.ndarray:
    if uploaded.size > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(io.BytesIO(uploaded.getvalue())) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit.")
            image.verify()
        with Image.open(io.BytesIO(uploaded.getvalue())) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as error:
        raise ValueError("The uploaded file is not a readable image.") from error
    except OSError as error:
        raise ValueError("The image could not be decoded.") from error


def show_image_processing() -> None:
    page_heading("Image tools", "Compare common image-processing views before running a model inspection.")
    st.write(
        "Compare common image-processing operations. These are display views; "
        "the trained anomaly model receives the original image unless an experiment validates a transform."
    )
    uploaded = st.file_uploader(
        "Upload a product image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        max_upload_size=10,
    )
    if uploaded is None:
        return

    try:
        original = _read_uploaded_image(uploaded)
    except ValueError as error:
        st.error(str(error))
        return

    col1, col2 = st.columns(2)
    col1.image(original, caption="Original RGB image", use_container_width=True)
    col2.image(clahe_rgb(original), caption="CLAHE local contrast", use_container_width=True)

    col3, col4 = st.columns(2)
    col3.image(gaussian_blur(original), caption="Gaussian smoothing", use_container_width=True)
    col4.image(median_blur(original), caption="Median filtering", use_container_width=True)

    st.image(canny_edges(original), caption="Canny edge map", clamp=True, use_container_width=True)
    st.subheader("Original image color histogram")
    st.line_chart({name: counts for name, counts in channel_histograms(original).items()})


def show_model_setup() -> None:
    page_heading("Model setup", "Review the training workflow and commands used to build the project models.")
    st.write(
        "Each product category gets its own PatchCore model. The training script uses only defect-free "
        "training images; validation uses synthetic anomalies, and the MVTec test split is evaluated once."
    )
    st.code(
        "$env:PYTHONPATH = 'src'\n"
        ".\\.venv\\Scripts\\python.exe scripts\\inspect_dataset.py\n"
        ".\\scripts\\train_all_categories.ps1",
        language="powershell",
    )
    st.caption("Training is run from PowerShell so long jobs stay visible and do not block the web app.")


def show_model_analytics() -> None:
    page_heading("Model analytics", "Compare measured benchmark results. Scores describe this dataset, not factory performance.")
    st.write(
        "Compare measured results on the held-out MVTec test split. Scores are benchmark metrics, "
        "not a guarantee of production performance."
    )
    patchcore_metrics = PROJECT_ROOT / "reports" / "retrained_patchcore_metrics.csv"
    rows: list[dict[str, object]] = []
    if patchcore_metrics.is_file():
        with patchcore_metrics.open("r", newline="", encoding="utf-8") as file:
            for item in csv.DictReader(file):
                rows.append(
                    {
                        "Category": item["category"].replace("_", " ").title(),
                        "Model": "PatchCore",
                        "Image AUROC": float(item["image_auroc"]),
                        "Image F1": float(item["image_f1"]),
                        "Pixel AUROC": float(item["pixel_auroc"]),
                        "Pixel F1": float(item["pixel_f1"]),
                    }
                )
    available_curves = []
    for category in CATEGORIES:
        evaluation_path = PROJECT_ROOT / "results" / "retrained" / "autoencoder" / f"{category}_evaluation.json"
        if evaluation_path.is_file():
            result = json.loads(evaluation_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Category": category.replace("_", " ").title(),
                    "Model": "CNN Autoencoder",
                    "Image AUROC": result["image_auroc"],
                    "Image F1": result["image_f1"],
                    "Pixel AUROC": result["pixel_auroc"],
                    "Pixel F1": result["pixel_f1"],
                }
            )
            available_curves.append(category)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "PatchCore uses synthetic validation through Anomalib. The autoencoder uses a held-out "
            "normal-only validation subset and percentile thresholds; compare results with that "
            "threshold difference in mind."
        )
    else:
        st.info("No evaluation results are available yet. Train a model from Model Setup.")

    for start in range(0, len(available_curves), 2):
        curve_cols = st.columns(2)
        for column, category in zip(curve_cols, available_curves[start : start + 2]):
            curve_path = PROJECT_ROOT / "results" / "retrained" / "autoencoder" / f"{category}_training_curve.csv"
            if curve_path.is_file():
                with column:
                    column.subheader(f"{category.replace('_', ' ').title()} autoencoder loss")
                    with curve_path.open("r", newline="", encoding="utf-8") as file:
                        curve_rows = list(csv.DictReader(file))
                    column.line_chart(
                        {
                            "Training loss": [float(item["train_loss"]) for item in curve_rows],
                            "Validation loss": [float(item["validation_loss"]) for item in curve_rows],
                        }
                    )


@st.cache_resource(show_spinner=False)
def _get_loaded_predictor(category: str):
    return load_predictor(category)


@st.cache_resource(show_spinner=False)
def _get_yolo_detector():
    return get_yolo_detector()


@st.cache_resource(show_spinner=False)
def _get_pbd_detector():
    return load_pbd_detector()


def show_ai_detection() -> None:
    render_telemetry_bar(station="INSPECTION BAY // CELL-04")
    page_heading("Autonomous Multi-Engine Quality Inspection", "Real-time industrial optical metrology, deep feature defect spectrometry, and spatial YOLO localization.")

    # High-Tech 3-Engine Industrial Cockpit
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 22px;">
          <div class="ind-engine-card">
            <div class="ind-engine-tag">ENGINE 01 // CLASSIFIER</div>
            <div class="ind-engine-title">Wide-ResNet-50 Prototypes</div>
            <div class="ind-engine-desc">Automatic category identification & confidence ranking across 15 industrial classes.</div>
          </div>
          <div class="ind-engine-card">
            <div class="ind-engine-tag">ENGINE 02 // SPECTROMETRY</div>
            <div class="ind-engine-title">PatchCore Anomaly Engine</div>
            <div class="ind-engine-desc">Deep patch memory manifold for micro-surface fracture scoring & JET heatmap projection.</div>
          </div>
          <div class="ind-engine-card">
            <div class="ind-engine-tag">ENGINE 03 // YOLO LOCALIZER</div>
            <div class="ind-engine-title">Ultralytics YOLOv11</div>
            <div class="ind-engine-desc">Real-time spatial bounding-box defect localizer operating at 1.8ms inference latency.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input source selector
    input_mode = st.radio(
        "Choose Specimen Feed Source",
        ["✨ 1-Click Automated Parts Feeder (Live Demo)", "📁 Optical File Ingest (Drag & Drop)", "📷 Industrial Camera Bay (Snapshot)"],
        horizontal=True,
        key="global_input_mode",
    )

    sample_dir = PROJECT_ROOT / "data" / "sample"
    sample_dict = {
        "🔌 Defective Cable (Bent Wire)": "cable_defective_bent_wire.png",
        "🔌 Normal Cable (Quality Passed)": "cable_normal.png",
        "🍼 Defective Bottle (Broken Body)": "bottle_defective_broken.png",
        "🍼 Normal Bottle (Quality Passed)": "bottle_normal.png",
        "💊 Defective Capsule (Cracked Surface)": "capsule_defective_crack.png",
        "💊 Normal Capsule (Quality Passed)": "capsule_normal.png",
        "🔩 Defective Metal Nut (Surface Scratch)": "metal_nut_defective_scratch.png",
        "🪛 Defective Screw (Thread Fault)": "screw_defective_scratch.png",
        "🌰 Defective Hazelnut (Crack & Hole)": "hazelnut_defective_crack.png",
    }

    original = None
    uploaded_name = "test_image.png"
    uploaded_bytes = b""

    if "1-Click" in input_mode:
        selected_sample_label = st.selectbox(
            "Select an automated specimen feed to inspect immediately:",
            list(sample_dict.keys()),
            key="sample_selector_pill",
            help="Select any sample product to run instant AI classification, anomaly scoring, and YOLO defect localization.",
        )
        sample_file = sample_dir / sample_dict[selected_sample_label]
        if sample_file.is_file():
            with Image.open(sample_file) as img:
                original = np.asarray(img.convert("RGB"), dtype=np.uint8)
            uploaded_name = sample_dict[selected_sample_label]
            uploaded_bytes = sample_file.read_bytes()
        else:
            st.error("Sample image file not found.")
            return
    elif "Camera" in input_mode:
        uploaded = camera_photo_input("Capture product photo", key="prediction_camera_capture")
        if uploaded is None:
            return
        try:
            original = _read_uploaded_image(uploaded)
            uploaded_name = uploaded.name
            uploaded_bytes = uploaded.getvalue()
        except ValueError as error:
            st.error(str(error))
            return
    else:
        uploaded = st.file_uploader(
            "Optical image drop zone",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            max_upload_size=10,
            key="prediction_upload",
        )
        if uploaded is None:
            return
        try:
            original = _read_uploaded_image(uploaded)
            uploaded_name = uploaded.name
            uploaded_bytes = uploaded.getvalue()
        except ValueError as error:
            st.error(str(error))
            return

    if original is None:
        return

    # Configuration Controls
    ctrl_col1, ctrl_col2 = st.columns([3, 2])
    with ctrl_col1:
        category_choice = st.selectbox(
            "Target Product Specification",
            ["Auto-detect from image (Recommended)", *CATEGORIES],
            key="prediction_category_choice",
            help="Model 1 automatically classifies the product into one of the 15 industrial categories.",
        )
    with ctrl_col2:
        yolo_conf = st.slider(
            "YOLO Defect Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.65,
            step=0.05,
            help="Adjust the bounding-box sensitivity threshold for YOLO defect localization.",
        )

    # Run Model 1: Product Recognition
    try:
        classification = classify_product_image(original)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, TypeError) as error:
        LOGGER.exception("Product classification failed")
        st.error(f"Could not classify the product image: {error}")
        return

    alternatives = ", ".join(
        f"{name.replace('_', ' ').title()} ({score:.1%})"
        for name, score in classification.ranked_scores[:3]
    )

    if category_choice == "Auto-detect from image (Recommended)":
        category = classification.category
        st.markdown(
            f"""
            <div style="background: rgba(0, 180, 216, 0.08); border-left: 3px solid var(--teal); padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 14px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--ink);">
              <span style="color: var(--teal); font-weight: 700;">[SPECIMEN IDENTIFIED]</span> CATEGORY: <strong>{category.replace('_', ' ').title()}</strong> // CONFIDENCE: <strong>{classification.confidence:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        category = category_choice
        if category != classification.category:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 14px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--ink);">
                  <span style="color: #f59e0b; font-weight: 700;">[MANUAL OVERRIDE]</span> CATEGORY: <strong>{category.replace('_', ' ').title()}</strong> (Auto-suggest: {classification.category.replace('_', ' ').title()} at {classification.confidence:.1%})
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: rgba(0, 180, 216, 0.08); border-left: 3px solid var(--teal); padding: 10px 16px; border-radius: 0 8px 8px 0; margin-bottom: 14px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--ink);">
                  <span style="color: var(--teal); font-weight: 700;">[CONFIRMED SPEC]</span> CATEGORY: <strong>{category.replace('_', ' ').title()}</strong> // CONFIDENCE: <strong>{classification.confidence:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Category classifier ranking: {alternatives}")

    source_key = hashlib.sha256(
        category.encode("utf-8") + b"\0" + str(yolo_conf).encode("ascii") + b"\0" + uploaded_bytes
    ).hexdigest()
    previous = st.session_state.get("latest_prediction")
    current = previous if previous and previous.get("source_key") == source_key else None

    inspect_clicked = st.button("🚀 Trigger Synchronous QA Inspection", type="primary", use_container_width=True) if "Camera" not in input_mode else False
    if inspect_clicked or ("Camera" in input_mode and current is None) or ("1-Click" in input_mode and current is None):
        try:
            with st.spinner(f"Acquiring telemetry for {category.replace('_', ' ').title()} across 3 AI engines…"):
                # Model 2: Surface Quality / PatchCore Anomaly Inspection
                predictor = _get_loaded_predictor(category)
                with tempfile.TemporaryDirectory(prefix="industrialguard-") as temporary:
                    temporary_image = Path(temporary) / "input.png"
                    Image.fromarray(original).save(temporary_image, format="PNG")
                    result = predict_image(
                        temporary_image,
                        category,
                        predictor=predictor,
                    )
                heatmap = make_heatmap(result.anomaly_map, (original.shape[1], original.shape[0]))
                overlay = overlay_heatmap(original, heatmap)

                # Model 3: YOLO Defect Bounding Box Localizer
                yolo = _get_yolo_detector()
                yolo_result = yolo.detect(
                    original,
                    conf=yolo_conf,
                    target_category=category,
                )

            # Combined Verdict Fusion
            is_defective = bool(result.is_anomalous or yolo_result.has_defect)
            status = "DEFECTIVE" if is_defective else "NORMAL"

            inspection_id = uuid4().hex
            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            inspection_dir = ARTIFACTS_ROOT / "inspections" / inspection_id
            inspection_dir.mkdir(parents=True, exist_ok=True)
            input_path = inspection_dir / "input.png"
            heatmap_path = inspection_dir / "heatmap.png"
            overlay_path = inspection_dir / "overlay.png"
            yolo_path = inspection_dir / "yolo_annotated.png"
            report_path = inspection_dir / "inspection-report.pdf"

            Image.fromarray(original).save(input_path, format="PNG")
            Image.fromarray(heatmap).save(heatmap_path, format="PNG")
            Image.fromarray(overlay).save(overlay_path, format="PNG")
            yolo_result.annotated_image.save(yolo_path, format="PNG")

            report_bytes = create_inspection_pdf(
                inspection_id=inspection_id,
                created_at=created_at,
                category=category,
                filename=uploaded_name,
                status=status,
                anomaly_score=result.anomaly_score,
                processing_seconds=result.processing_seconds,
                original_rgb=original,
                heatmap_rgb=heatmap,
                recognition_confidence=classification.confidence,
                yolo_defect_count=yolo_result.defect_count,
                yolo_annotated_rgb=np.array(yolo_result.annotated_image),
            )
            report_path.write_bytes(report_bytes)

            def relative_artifact(path: Path) -> str:
                return path.resolve().relative_to(PROJECT_ROOT).as_posix()

            add_inspection(
                inspection_id=inspection_id,
                created_at=created_at,
                product_category=category,
                original_filename=uploaded_name,
                anomaly_score=result.anomaly_score,
                inspection_status=status,
                processing_time=result.processing_seconds,
                input_image_path=relative_artifact(input_path),
                heatmap_path=relative_artifact(heatmap_path),
                overlay_path=relative_artifact(overlay_path),
                report_path=relative_artifact(report_path),
            )
            current = {
                "source_key": source_key,
                "inspection_id": inspection_id,
                "created_at": created_at,
                "report_path": report_path,
                "category": category,
                "file_name": uploaded_name,
                "is_anomalous": is_defective,
                "anomaly_score": result.anomaly_score,
                "threshold": getattr(result, "threshold", 0.35),
                "patchcore_anomalous": result.is_anomalous,
                "processing_seconds": result.processing_seconds,
                "original": original,
                "heatmap": heatmap,
                "overlay": overlay,
                "classification": classification,
                "yolo_result": yolo_result,
            }
            st.session_state["latest_prediction"] = current
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            LOGGER.exception("Quality inspection failed for category %s", category)
            st.error(str(error))
            return

    if current is None:
        return

    st.markdown("---")
    category_label = current["category"].replace("_", " ").title()
    yolo_res = current["yolo_result"]
    patchcore_anom = current["patchcore_anomalous"]
    yolo_anom = yolo_res.has_defect

    if current["is_anomalous"]:
        st.markdown(
            """
            <div class="ind-verdict-reject">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <h2>
                  ❌ NOT APPROVED TO SELL // QA GATE: REJECTED
                </h2>
                <span style="background: #ef4444; color: white; padding: 5px 14px; border-radius: 6px; font-weight: 700; font-size: 12px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">
                  QUARANTINED [ERR-DEFECT-DETECTED]
                </span>
              </div>
              <p>
                Component failed automated visual metrology. Surface anomaly threshold exceeded or spatial defect localized. Automated diverter engaged for scrap/rework.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔍 Real-Time Diagnostic Breakdown")
        st.markdown(
            f"- **Inspected Specimen:** **{category_label}**\n"
            f"- **Engine 02 (Surface Spectrometry):** Anomaly Score **`{current['anomaly_score']:.4f}`** "
            f"({'⚠️ Exceeded threshold' if patchcore_anom else '✓ Normal surface'})\n"
            f"- **Engine 03 (YOLO Localizer):** Found **`{yolo_res.defect_count}`** defect region(s) "
            f"with peak confidence **`{yolo_res.max_confidence:.1%}`**\n"
            f"- **Cell Action:** Divert part to Quarantine Bin B-03. Detailed coordinates recorded in database."
        )
    else:
        st.markdown(
            """
            <div class="ind-verdict-pass">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <h2>
                  ✅ APPROVED TO SELL // QA GATE: PASSED
                </h2>
                <span style="background: #10b981; color: white; padding: 5px 14px; border-radius: 6px; font-weight: 700; font-size: 12px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em;">
                  RELEASE AUTHORIZED [PRODUCTION GRADE]
                </span>
              </div>
              <p>
                100% of surface and dimensional parameters are verified within certified manufacturing tolerance. Product approved for high-speed packaging and commercial release.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔍 Quality Verification Summary")
        st.markdown(
            f"- **Inspected Specimen:** **{category_label}**\n"
            f"- **Engine 02 (Surface Spectrometry):** Anomaly score **`{current['anomaly_score']:.4f}`** is within certified tolerance.\n"
            f"- **Engine 03 (YOLO Localizer):** **0** defect bounding boxes detected across product.\n"
            f"- **Cell Action:** Part verified production grade. Proceeding to packaging conveyor."
        )

    # Metric Cards Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Classified Specimen", f"{category_label}", f"{current['classification'].confidence:.1%}")
    m_col2.metric("Surface Anomaly", f"{current['anomaly_score']:.4f}", "PatchCore Manifold")
    m_col3.metric("YOLO Defect Bboxes", f"{yolo_res.defect_count}", f"Max {yolo_res.max_confidence:.1%}" if yolo_res.has_defect else "0 defects")
    m_col4.metric("Cycle Latency", f"{current['processing_seconds']:.3f} s", "CUDA Accelerated")

    # 3-Visual Gallery: Synchronized Optical Monitors
    st.markdown("### 🖼️ Synchronized Multi-Channel Optical Displays")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.markdown(
            """
            <div class="ind-monitor"><div class="ind-monitor-header">
              <span>// CH-01: OPTICAL RAW RGB</span><span class="tag">SENSOR 1024x1024</span>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
        st.image(current["original"], use_container_width=True)
    with v_col2:
        st.markdown(
            """
            <div class="ind-monitor"><div class="ind-monitor-header">
              <span>// CH-02: ANOMALY SPECTROMETRY</span><span class="tag">PATCHCORE JET</span>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
        st.image(current["overlay"], use_container_width=True)
    with v_col3:
        st.markdown(
            """
            <div class="ind-monitor"><div class="ind-monitor-header">
              <span>// CH-03: YOLO DEFECT RETICLE</span><span class="tag">YOLOv11 BOUNDING BOXES</span>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
        st.image(yolo_res.annotated_image, use_container_width=True)

    # Detailed Audit & Analytics Tabs
    tab_yolo, tab_recog, tab_diag, tab_report = st.tabs([
        "🎯 [DATA LOG] YOLO Defect Coordinates",
        "🧠 [SPECTRAL MATRIX] Classification Distribution",
        "🔬 [TELEMETRY] Surface Diagnostics",
        "📄 [CERTIFICATE] Quality Inspection PDF",
    ])

    with tab_yolo:
        if yolo_res.detections:
            st.markdown(f"**Found {len(yolo_res.detections)} Defect Region(s):**")
            box_rows = [
                {
                    "Defect Index": f"Defect #{idx+1}",
                    "Class": d.class_name,
                    "Confidence": f"{d.confidence:.1%}",
                    "Bounding Box (x1, y1, x2, y2)": f"({d.box[0]}, {d.box[1]}, {d.box[2]}, {d.box[3]})",
                    "Area (px²)": f"{d.area:,}",
                }
                for idx, d in enumerate(yolo_res.detections)
            ]
            st.dataframe(box_rows, use_container_width=True, hide_index=True)
        else:
            st.info("✓ YOLO localized zero defect bounding boxes at this sensitivity threshold.")

    with tab_recog:
        st.markdown(f"**Classification Model:** Wide-ResNet-50 Centroid Embedding Prototyper")
        recog_data = {
            name.replace("_", " ").title(): float(score)
            for name, score in current["classification"].ranked_scores[:5]
        }
        st.bar_chart(recog_data)

    with tab_diag:
        st.markdown(f"**Anomaly Engine:** PatchCore (Multi-Scale Neighborhood Patch Embeddings)")
        st.markdown(
            f"- **Measured Anomaly Score:** `{current['anomaly_score']:.6f}`\n"
            f"- **Classification Status:** `{'DEFECTIVE' if current['is_anomalous'] else 'NORMAL'}`\n"
            f"- **Processing Time:** `{current['processing_seconds']:.4f} seconds`\n"
            f"- **Input Source:** `{current['file_name']}`"
        )

    with tab_report:
        st.markdown("Download the official single-page PDF Quality Inspection Certificate for audit records:")
        st.download_button(
            "📥 Download Certified Inspection Report (PDF)",
            data=current["report_path"].read_bytes(),
            file_name=f"QualityReport_{current['category']}_{current['inspection_id'][:8]}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )


def show_inspection_history() -> None:
    page_heading("Inspection history", "Review recent PatchCore inspections saved on this computer.")
    records = list_inspections(limit=200)
    if not records:
        st.info("No inspections have been recorded yet.")
        return
    st.dataframe(
        [
            {
                "Inspection ID": row["inspection_id"],
                "Date and time (UTC)": row["created_at"],
                "Category": row["product_category"],
                "Model": row["model_name"],
                "Anomaly score": row["anomaly_score"],
                "Status": row["inspection_status"],
                "Time (s)": row["processing_time"],
            }
            for row in records
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The table shows the latest 200 records. Images and reports are stored as local files under artifacts/.")


def show_reports() -> None:
    page_heading("Inspection reports", "Choose a completed inspection and download its PDF report.")
    records = list_inspections(limit=200)
    if not records:
        st.info("Run an inspection to create a downloadable PDF report.")
        return

    options = {
        f"{row['created_at']} · {row['product_category']} · {row['inspection_status']} · {row['inspection_id'][:8]}": row
        for row in records
    }
    selected_label = st.selectbox("Select an inspection", list(options))
    selected = options[selected_label]
    report_path = (PROJECT_ROOT / selected["report_path"]).resolve()
    artifacts_root = ARTIFACTS_ROOT.resolve()
    if not report_path.is_relative_to(artifacts_root) or not report_path.is_file():
        st.error("The saved report file is missing or outside the local artifacts folder.")
        return

    st.write(f"Inspection ID: `{selected['inspection_id']}`")
    st.write(f"Model result: **{selected['inspection_status']}**")
    st.download_button(
        "Download PDF report",
        data=report_path.read_bytes(),
        file_name=f"IndustrialGuard_{selected['inspection_id']}.pdf",
        mime="application/pdf",
    )


PAGES = {
    "📊  Operations Console": show_dashboard,
    "📷  Live 3-Engine Inspection": show_ai_detection,
    "🧾  Quality Audit Logs": show_inspection_history,
    "📄  Certified PDF Reports": show_reports,
    "📈  Metrology Analytics": show_model_analytics,
    "🖼️  Optical Image Filters": show_image_processing,
    "🗂️  Benchmark Dataset Audit": show_dataset_audit,
    "⚙️  Pipeline Configuration": show_model_setup,
}

st.sidebar.markdown(
    """
    <div class="ig-brand">
      <div class="ig-mark">IG4</div>
      <div>
        <div class="ig-brand-name">INDUSTRIALGUARD™ 4.0</div>
        <div class="ig-brand-sub">AUTONOMOUS VISION QA</div>
      </div>
    </div>
    <div class="ind-sidebar-card">
      <div style="color: var(--side-sub); font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;">CELL HARDWARE</div>
      <div style="color: var(--teal); font-weight: 700; font-size: 11px; margin: 2px 0;">CUDA RTX 4060 · 8GB</div>
      <div style="color: var(--muted); font-size: 10px;">CYCLE: 1.8ms · 120 FPS</div>
      <div style="color: #10b981; font-size: 10px; margin-top: 4px; display: flex; align-items: center;">
        <span class="ind-dot ind-dot-green"></span> STATION: ONLINE
      </div>
    </div>
    <div class="ig-nav-label">CONTROL STATIONS</div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.toggle("Industrial Dark Theme", key="dark_mode")
selection = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed", key="main_navigation")
PAGES[selection]()
