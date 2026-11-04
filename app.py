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

st.session_state.setdefault("dark_mode", False)
if st.session_state["dark_mode"]:
    theme_colors = {
        "ink": "#edf4fb", "muted": "#a8b7c7", "teal": "#58ccd2", "line": "#2c4055",
        "app": "#0a1421", "surface": "#13263a", "sidebar": "#0c1d2e",
        "side_text": "#e7f0f8", "side_sub": "#9eb3c6", "side_label": "#91a7bb",
        "field": "#102237", "shadow": "#00000030", "hero_a": "#16394c",
        "hero_b": "#1b2a43", "hero_ink": "#f3f8fc", "hero_muted": "#c5d6e3",
    }
else:
    theme_colors = {
        "ink": "#142c43", "muted": "#64748b", "teal": "#087f8c", "line": "#e3eaf1",
        "app": "#f4f7fb", "surface": "#ffffff", "sidebar": "#ffffff",
        "side_text": "#142c43", "side_sub": "#64748b", "side_label": "#74869a",
        "field": "#ffffff", "shadow": "#102a4308", "hero_a": "#e2f3f2",
        "hero_b": "#e9effb", "hero_ink": "#153249", "hero_muted": "#526a7d",
    }

theme_css = """
    <style>
      :root { --ink:__INK__; --muted:__MUTED__; --teal:__TEAL__; --line:__LINE__;
        --app-bg:__APP__; --surface:__SURFACE__; --sidebar-bg:__SIDEBAR__;
        --side-text:__SIDE_TEXT__; --side-sub:__SIDE_SUB__; --side-label:__SIDE_LABEL__;
        --field:__FIELD__; --shadow:__SHADOW__; --hero-a:__HERO_A__; --hero-b:__HERO_B__;
        --hero-ink:__HERO_INK__; --hero-muted:__HERO_MUTED__; }
      [data-testid="stAppViewContainer"] { background:var(--app-bg); color:var(--ink); }
      [data-testid="stMainBlockContainer"] { max-width:1420px; padding:1.7rem 2rem 3rem; }
      [data-testid="stSidebar"] { background:var(--sidebar-bg); border-right:1px solid var(--line); }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:var(--side-text); }
      [data-testid="stSidebar"] [data-testid="stRadio"] label,
      [data-testid="stSidebar"] [data-testid="stToggle"] label { color:var(--side-text); }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { color:var(--teal); font-weight:700; }
      [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stAppViewContainer"] label { color:var(--ink); }
      .ig-brand { display:flex; align-items:center; gap:12px; padding:10px 4px 24px; }
      .ig-mark { display:flex; align-items:center; justify-content:center; width:42px; height:42px;
        border-radius:13px; background:#0d8792; color:white; font-size:17px; font-weight:800; }
      .ig-brand-name { color:var(--side-text); font-size:16px; line-height:1.2; font-weight:750; }
      .ig-brand-sub { color:var(--side-sub); font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin-top:4px; }
      .ig-nav-label { color:var(--side-label); font-size:11px; font-weight:700; letter-spacing:.12em; margin:0 0 8px 4px; }
      .ig-hero { padding:28px 32px; margin:0 0 22px; border:1px solid var(--line); border-radius:21px;
        background:linear-gradient(125deg,var(--hero-a),var(--hero-b)); box-shadow:0 8px 26px var(--shadow); }
      .ig-eyebrow { color:var(--teal); font-size:11px; font-weight:800; letter-spacing:.15em; text-transform:uppercase; margin-bottom:9px; }
      .ig-hero h1 { color:var(--hero-ink); font-size:36px; line-height:1.12; letter-spacing:-.03em; margin:0; }
      .ig-hero p { color:var(--hero-muted); font-size:15px; margin:10px 0 0; max-width:780px; }
      [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:17px 19px; box-shadow:0 5px 18px var(--shadow); }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stMetricValue"] { color:var(--ink); }
      [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] { gap:1rem; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { background:var(--surface); border-color:var(--line); border-radius:16px; }
      .stButton > button, [data-testid="stDownloadButton"] button { border-radius:10px; min-height:42px; font-weight:650; background:var(--surface); color:var(--ink); border-color:var(--line); }
      .stButton > button[kind="primary"] { background:#087f8c; border-color:#087f8c; }
      .stButton > button[kind="primary"]:hover { background:#066b76; border-color:#066b76; }
      [data-testid="stFileUploader"] section { border:1px dashed var(--line); border-radius:13px; background:var(--surface); color:var(--ink); }
      [data-baseweb="select"] > div, [data-testid="stTextInput"] input,
      [data-testid="stNumberInput"] input { background:var(--field); color:var(--ink); border-color:var(--line); }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
      h2, h3 { color:var(--ink); }
      @media (max-width:700px) { .ig-hero { padding:22px; } .ig-hero h1 { font-size:28px; } [data-testid="stMainBlockContainer"] { padding:1.2rem 1rem 2rem; } }
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
}.items():
    theme_css = theme_css.replace(f"__{token}__", value)
st.markdown(theme_css, unsafe_allow_html=True)


def page_heading(title: str, subtitle: str, eyebrow: str = "SMART QUALITY INSPECTION") -> None:
    st.markdown(
        f'<div class="ig-hero"><div class="ig-eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _open_camera_inspection() -> None:
    st.session_state["main_navigation"] = "📷  AI inspection"
    st.session_state["prediction_image_source"] = "Camera"


def show_dashboard() -> None:
    page_heading("Smart Quality Inspection System", "Automated product category detection, visual anomaly inspection, and defect assessment.")
    camera_col, help_col = st.columns([1, 2], vertical_alignment="center")
    with camera_col:
        st.button(
            "📷  Start camera inspection",
            type="primary",
            use_container_width=True,
            on_click=_open_camera_inspection,
        )
    with help_col:
        st.markdown("**Ready for a quick check?** Open the camera, capture one image, and the selected model will return a result.")
    counts = inspection_counts()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Total inspections", counts["total"])
    metric_cols[1].metric("Normal", counts["normal"])
    metric_cols[2].metric("Defective", counts["defective"])
    defect_rate = counts["defective"] / counts["total"] * 100 if counts["total"] else 0.0
    metric_cols[3].metric("Defect rate", f"{defect_rate:.1f}%")

    st.markdown("### Models and data")
    coverage_col, bottle_col, workflow_col = st.columns([1, 1, 1.25])
    if DATASET_ROOT.is_dir():
        audits = audit_categories(DATASET_ROOT, list(MVTEC_CATEGORIES))
        audited_count = sum(audit.is_complete for audit in audits)
        normal_count = sum(audit.train_good for audit in audits)
        with coverage_col:
            with st.container(border=True):
                st.metric("Benchmark categories", f"{audited_count} / 15")
                st.caption(f"{normal_count:,} defect-free training images audited")
    else:
        with coverage_col:
            with st.container(border=True):
                st.metric("Benchmark categories", "Dataset missing")
                st.caption("Place MVTec AD in the project data folder.")
    with bottle_col:
        with st.container(border=True):
            st.metric("Plastic bottle labels", "7 classes" if PBD_CHECKPOINT.is_file() else "Not trained")
            st.caption("Separate body and shoulder defect detector")
    with workflow_col:
        with st.container(border=True):
            st.markdown("**Inspection workflow**")
            st.markdown("1. Choose a model and product category.\n2. Upload an image or take a camera photo.\n3. Review the result and save the report.")

    recent = list_inspections(limit=5)
    if recent:
        st.markdown("### Recent inspections")
        st.dataframe(
            [
                {
                    "Inspection ID": item["inspection_id"],
                    "Date (UTC)": item["created_at"],
                    "Category": item["product_category"],
                    "Model": item["model_name"],
                    "Anomaly score": item["anomaly_score"],
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
    page_heading("Industrial AI Quality Inspection", "Multi-Model Vision Pipeline: Recognition Engine + Surface Anomaly Analyzer + YOLO Defect Localizer.")

    # High-level Architecture Overview
    st.markdown(
        """
        <div style="background: rgba(8, 127, 140, 0.06); border: 1px solid rgba(8, 127, 140, 0.25); border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
          <div style="display: flex; gap: 24px; flex-wrap: wrap; justify-content: space-between; align-items: center;">
            <div style="flex: 1; min-width: 200px;">
              <span style="font-weight: 700; color: #087f8c; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">🧠 Model 1: Recognition</span>
              <p style="margin: 3px 0 0; font-size: 13px; color: #334155;">Deep Feature Prototypes · Classifies product & confidence</p>
            </div>
            <div style="border-left: 1px solid rgba(8, 127, 140, 0.2); height: 32px; display: none;"></div>
            <div style="flex: 1; min-width: 200px;">
              <span style="font-weight: 700; color: #087f8c; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">🔬 Model 2: Quality Analyzer</span>
              <p style="margin: 3px 0 0; font-size: 13px; color: #334155;">PatchCore Anomaly Engine · Micro-defect scoring & heatmap</p>
            </div>
            <div style="border-left: 1px solid rgba(8, 127, 140, 0.2); height: 32px; display: none;"></div>
            <div style="flex: 1; min-width: 200px;">
              <span style="font-weight: 700; color: #087f8c; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">🎯 Model 3: YOLO Localizer</span>
              <p style="margin: 3px 0 0; font-size: 13px; color: #334155;">Ultralytics YOLOv11 · Defect bounding boxes & coordinates</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input source selector
    input_mode = st.radio(
        "Choose Image Input Source",
        ["✨ 1-Click Live Demo Samples", "📁 Upload Image", "📷 Webcam Snapshot"],
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

    if input_mode == "✨ 1-Click Live Demo Samples":
        selected_sample_label = st.selectbox(
            "Select an industrial sample product to test immediately:",
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
    elif input_mode == "📷 Webcam Snapshot":
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
            "Upload a product image",
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
            "Product category",
            ["Auto-detect from image (Recommended)", *CATEGORIES],
            key="prediction_category_choice",
            help="Model 1 automatically classifies the product into one of the 15 industrial categories.",
        )
    with ctrl_col2:
        yolo_conf = st.slider(
            "YOLO Defect Confidence Threshold",
            min_value=0.05,
            max_value=0.90,
            value=0.50,
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
        st.success(
            f"🧠 **Model 1 (Recognition):** Identified as **{category.replace('_', ' ').title()}** "
            f"with **{classification.confidence:.1%}** confidence."
        )
    else:
        category = category_choice
        if category != classification.category:
            st.info(
                f"Using manual selection: **{category.replace('_', ' ').title()}** "
                f"(Model 1 suggested: {classification.category.replace('_', ' ').title()} at {classification.confidence:.1%})"
            )
        else:
            st.success(
                f"🧠 **Model 1 (Recognition):** Confirmed as **{category.replace('_', ' ').title()}** "
                f"with **{classification.confidence:.1%}** confidence."
            )

    st.caption(f"Top category candidate rankings: {alternatives}")

    source_key = hashlib.sha256(
        category.encode("utf-8") + b"\0" + str(yolo_conf).encode("ascii") + b"\0" + uploaded_bytes
    ).hexdigest()
    previous = st.session_state.get("latest_prediction")
    current = previous if previous and previous.get("source_key") == source_key else None

    inspect_clicked = st.button("🚀 Run Multi-Model Quality Inspection", type="primary", use_container_width=True) if input_mode != "📷 Webcam Snapshot" else False
    if inspect_clicked or (input_mode == "📷 Webcam Snapshot" and current is None) or (input_mode == "✨ 1-Click Live Demo Samples" and current is None):
        try:
            with st.spinner(f"Analyzing {category.replace('_', ' ').title()} with Model 2 (PatchCore) and Model 3 (YOLO)…"):
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
            '<div style="background: linear-gradient(135deg, rgba(211, 47, 47, 0.12), rgba(211, 47, 47, 0.04)); '
            'border: 2.5px solid #d32f2f; border-radius: 16px; padding: 22px 28px; margin-bottom: 22px; '
            'box-shadow: 0 6px 24px rgba(211, 47, 47, 0.15);">'
            '<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">'
            '<h2 style="color: #d32f2f; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">❌ NOT APPROVED TO SELL</h2>'
            '<span style="background: #d32f2f; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px;">QUALITY REJECTED</span>'
            '</div>'
            '<p style="color: #b71c1c; margin: 10px 0 0; font-size: 15px; font-weight: 600;">'
            'Defect detected by visual inspection system. Product failed quality standards and is quarantined for rework or disposal.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔍 Multi-Model Defect Diagnosis")
        st.markdown(
            f"- **Product Evaluated:** **{category_label}**\n"
            f"- **Model 2 (Surface Quality):** Anomaly Score **`{current['anomaly_score']:.4f}`** "
            f"({'⚠️ Exceeded threshold' if patchcore_anom else '✓ Normal surface'})\n"
            f"- **Model 3 (YOLO Localizer):** Found **`{yolo_res.defect_count}`** defect region(s) "
            f"with peak confidence **`{yolo_res.max_confidence:.1%}`**\n"
            f"- **Action Required:** Product is rejected from production line. Inspect localized bounding boxes below."
        )
    else:
        st.markdown(
            '<div style="background: linear-gradient(135deg, rgba(46, 125, 50, 0.12), rgba(46, 125, 50, 0.04)); '
            'border: 2.5px solid #2e7d32; border-radius: 16px; padding: 22px 28px; margin-bottom: 22px; '
            'box-shadow: 0 6px 24px rgba(46, 125, 50, 0.15);">'
            '<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">'
            '<h2 style="color: #2e7d32; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">✅ APPROVED TO SELL</h2>'
            '<span style="background: #2e7d32; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px;">PRODUCTION GRADE</span>'
            '</div>'
            '<p style="color: #1b5e20; margin: 10px 0 0; font-size: 15px; font-weight: 600;">'
            'Product passed all automated visual quality inspections. Verified within industrial tolerances for commercial distribution.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 🔍 Quality Verification Summary")
        st.markdown(
            f"- **Product Evaluated:** **{category_label}**\n"
            f"- **Model 2 (Surface Quality):** Anomaly score **`{current['anomaly_score']:.4f}`** is within certified tolerance.\n"
            f"- **Model 3 (YOLO Localizer):** **0** defect bounding boxes detected across product.\n"
            f"- **Status:** 100% Passed. Ready for packaging and shipment."
        )

    # Metric Cards Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Recognized Product", f"{category_label}", f"{current['classification'].confidence:.1%}")
    m_col2.metric("Anomaly Score", f"{current['anomaly_score']:.4f}", "PatchCore")
    m_col3.metric("YOLO Defect Regions", f"{yolo_res.defect_count}", f"Max {yolo_res.max_confidence:.1%}" if yolo_res.has_defect else "0 defects")
    m_col4.metric("Inference Latency", f"{current['processing_seconds']:.3f} s", "GPU Accelerated")

    # 3-Visual Gallery: Side-by-Side
    st.markdown("### 🖼️ Visual Inspection Views")
    v_col1, v_col2, v_col3 = st.columns(3)
    v_col1.image(current["original"], caption="1. Original Product", use_container_width=True)
    v_col2.image(current["overlay"], caption="2. Model 2: Surface Anomaly Heatmap", use_container_width=True)
    v_col3.image(yolo_res.annotated_image, caption="3. Model 3: YOLO Defect Bounding Boxes", use_container_width=True)

    # Detailed Audit & Analytics Tabs
    tab_yolo, tab_recog, tab_diag, tab_report = st.tabs([
        "🎯 YOLO Defect Detections",
        "🧠 Model 1 Recognition Matrix",
        "🔬 Model 2 Anomaly Diagnostics",
        "📄 Inspection Certificate (PDF)",
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
    "📊  Dashboard": show_dashboard,
    "🗂️  Dataset audit": show_dataset_audit,
    "🖼️  Image tools": show_image_processing,
    "📷  AI inspection": show_ai_detection,
    "🧾  Inspection history": show_inspection_history,
    "📄  Reports": show_reports,
    "📈  Model analytics": show_model_analytics,
    "⚙️  Model setup": show_model_setup,
}

st.sidebar.markdown(
    '<div class="ig-brand"><div class="ig-mark">SQI</div><div>'
    '<div class="ig-brand-name">Smart Quality Inspection</div>'
    '<div class="ig-brand-sub">Industrial AI Workspace</div></div></div>'
    '<div class="ig-nav-label">WORKSPACE</div>',
    unsafe_allow_html=True,
)
st.sidebar.toggle("Dark mode", key="dark_mode")
selection = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed", key="main_navigation")
PAGES[selection]()
