"""SIGAP: photo-first food freshness screening plus legacy inventory analysis."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.local_vision import DEFAULT_MODEL_PATH, load_local_model, predict_photo
from src.predictor import (
    MODEL_PATH as INVENTORY_MODEL_PATH,
    predict_batch,
    predict_food_waste,
)

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "assets" / "styles.css"
INVENTORY_CSV_PATH = BASE_DIR / "data" / "data_penjualan.csv"
INVENTORY_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
TREE_IMAGE_PATH = BASE_DIR / "images" / "decision_tree.png"
PHOTO_METRICS_PATH = BASE_DIR / "models" / "metrics.json"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="SIGAP · Food Analysis",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand"><span class="brand-icon">S</span><span>SIGAP</span></div>
          <div class="local-pill"><span class="status-dot"></span> Local AI · No API</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_inventory_sample() -> pd.DataFrame:
    if not INVENTORY_CSV_PATH.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(INVENTORY_CSV_PATH)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_cached_photo_model():
    return load_local_model()


@st.cache_resource(show_spinner=False)
def get_cached_inventory_model_bundle():
    from src.predictor import load_model_bundle
    return load_model_bundle()


def render_photo_page() -> None:
    st.markdown('<h1 class="page-title">Fruit Freshness Check</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a photo to screen for visible signs of spoilage.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a fruit photo",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="JPG, PNG, or WEBP · Max 8 MB.",
        key="photo_upload",
    )

    image_bytes: bytes | None = None
    image_valid = False
    image_dimensions: tuple[int, int] | None = None
    image_digest: str | None = None

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            st.error("Image size must be 8 MB or less.")
        else:
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                    if width * height > MAX_IMAGE_PIXELS:
                        st.error("Image resolution is too large. Choose a smaller image.")
                    else:
                        image_dimensions = (width, height)
                        image_valid = True
            except (UnidentifiedImageError, OSError):
                st.error("This image could not be read. Try another JPG, PNG, or WEBP file.")

    if "photo_result_digest" not in st.session_state:
        st.session_state.photo_result_digest = None
    if "photo_result" not in st.session_state:
        st.session_state.photo_result = None
    if st.session_state.photo_result_digest != image_digest:
        st.session_state.photo_result_digest = image_digest
        st.session_state.photo_result = None

    photo_model_exists = DEFAULT_MODEL_PATH.is_file()
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        if image_valid and image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                st.image(image.convert("RGB"), use_container_width=True)
            meta_left, meta_right = st.columns([1, 1])
            with meta_left:
                st.caption("Selected photo")
            with meta_right:
                if image_dimensions:
                    st.caption(f"{image_dimensions[0]} × {image_dimensions[1]} px", help="Image resolution")
        else:
            st.markdown(
                """
                <div class="empty-preview">
                  <div class="empty-icon">↥</div>
                  <div class="empty-title">Photo preview</div>
                  <div class="empty-description">Choose a fruit photo to get started.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="panel-kicker">PHOTO CHECK</div>', unsafe_allow_html=True)
        if image_valid:
            st.markdown('<div class="panel-title">Photo ready to analyze</div>', unsafe_allow_html=True)
            if uploaded:
                st.caption(uploaded.name)
        else:
            st.markdown('<div class="panel-title">Start with a photo</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-copy">Make sure the fruit is clearly visible and well lit.</div>', unsafe_allow_html=True)

        if photo_model_exists:
            st.markdown('<div class="model-status"><span class="status-dot"></span> Photo model available</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="model-status model-missing"><span class="status-dot"></span> Photo model not available</div>', unsafe_allow_html=True)

        has_result = st.session_state.photo_result is not None
        can_analyze = bool(image_valid and image_bytes and photo_model_exists)
        if st.button("Analyze again" if has_result else "Analyze photo", type="primary", use_container_width=True, disabled=not can_analyze, key="analyze_photo_button"):
            try:
                with st.spinner("Analyzing photo locally…"):
                    st.session_state.photo_result = predict_photo(image_bytes, get_cached_photo_model())
            except Exception as exc:
                st.session_state.photo_result = None
                st.error("Photo analysis failed. Check the SIGAP model checkpoint.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

        if not photo_model_exists:
            with st.expander("Set up photo model"):
                st.write("The image dataset must be available, and the model must be trained once.")
                st.code("python scripts/download_dataset.py\npython scripts/train_model.py", language="powershell")

    result = st.session_state.photo_result
    if result:
        kind = result.get("assessment", {}).get("kind", "cannot_assess")
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        if kind == "possible_spoilage":
            title, result_class, icon = "Possible Signs of Spoilage", "result-warning", "!"
            advice = "Inspect the fruit directly. If you see mold or clear damage, do not eat it. Do not taste it to test."
        elif kind == "fresh_prediction":
            title, result_class, icon = "Predicted Fresh", "result-success", "✓"
            advice = "The model found no strong visual signs of spoilage. Check the smell, texture, and storage conditions before eating."
        else:
            title, result_class, icon = "Unable to Assess", "result-neutral", "?"
            advice = "Retake the photo with the fruit clearly visible, in focus, and well lit."

        st.markdown('<div class="section-heading">Screening Result</div>', unsafe_allow_html=True)
        with st.container(border=True):
            result_col, score_col = st.columns([1.6, 0.7], gap="large")
            with result_col:
                st.markdown(
                    f'<div class="result-heading {result_class}"><span class="result-icon">{icon}</span><div><div class="result-eyebrow">MODEL OUTPUT</div><div class="result-title">{title}</div></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="result-description">{advice}</div>', unsafe_allow_html=True)
            with score_col:
                st.markdown('<div class="score-label">Model Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-number">{score:.0%}</div>', unsafe_allow_html=True)
                st.progress(score)
                st.caption("Not a calibrated food-safety probability")

        st.caption("SIGAP assesses visual appearance only. This result does not guarantee that the food is safe to eat.")


def prepare_batch_results_for_display(results: pd.DataFrame) -> pd.DataFrame:
    """Translate legacy model fields for English UI without changing model schema."""
    display = results.copy()
    if "status" in display.columns:
        display["status"] = display["status"].map({
            "Potential Waste": "Potential Waste",
            "Safe": "Lower Risk",
            "Not evaluated": "Not Evaluated",
        }).fillna(display["status"])
    if "risk_level" in display.columns:
        display["risk_level"] = display["risk_level"].map({
            "HIGH RISK": "High Risk",
            "LOW RISK": "Low Risk",
            "—": "—",
        }).fillna(display["risk_level"])
    display = display.rename(columns={
        "product_name": "Product", "source_row": "CSV Row", "status": "Status",
        "risk_level": "Risk Level", "stok": "Stock", "terjual": "Sold",
        "sisa_stok": "Remaining Stock", "sisa_persen": "Remaining (%)",
        "model_confidence": "Model Score", "recommendation": "Recommendation",
        "validation_status": "Validation", "validation_error": "Validation Message",
    })
    return display


def prepare_inventory_data_for_display(data: pd.DataFrame) -> pd.DataFrame:
    """Expose English column names while preserving the legacy source file unchanged."""
    display = data.copy()
    display = display.rename(columns={
        "product_name": "Product", "stok": "Stock", "terjual": "Sold",
        "hari_ke": "Day Index", "cuaca": "Weather Code", "hari_besar": "Special Day",
        "sisa_persen": "Remaining (%)", "label": "Label",
    })
    if "Label" in display.columns:
        display["Label"] = display["Label"].map({0: "Safe", 1: "Potential Waste", 0.0: "Safe", 1.0: "Potential Waste"}).fillna(display["Label"])
    return display


def prepare_inventory_csv_for_download(data: pd.DataFrame) -> pd.DataFrame:
    """Export the sample with English headers; predictor accepts these aliases."""
    return data.rename(columns={
        "stok": "stock", "terjual": "sold", "hari_ke": "day_index",
        "cuaca": "weather_code", "hari_besar": "special_day",
        "sisa_persen": "remaining_percentage", "label": "label",
    })


def render_inventory_page() -> None:
    st.markdown('<h1 class="page-title">Inventory Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Use the existing Decision Tree model to assess leftover-stock risk.</p>', unsafe_allow_html=True)

    sample = load_inventory_sample()
    model_exists = INVENTORY_MODEL_PATH.is_file()
    if not sample.empty:
        total = int(len(sample))
        potential = int((pd.to_numeric(sample.get("label"), errors="coerce") == 1).sum())
        safe = int((pd.to_numeric(sample.get("label"), errors="coerce") == 0).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Sample Rows", f"{total}", help="Example rows included with the project.")
        k2.metric("Potential Waste", f"{potential}")
        k3.metric("Lower Risk", f"{safe}")
    else:
        st.info("The sample dataset was not found. You can still use manual input if the model is available.")

    with st.container(border=True):
        st.markdown("#### Check One Product")
        with st.form("single_inventory_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                product_name = st.text_input("Product name", value="Sample product", max_chars=80)
                stock = st.number_input("Stock quantity", min_value=1, max_value=100000, value=100, step=1)
            with c2:
                sold = st.number_input("Units sold", min_value=0, max_value=int(stock), value=min(65, int(stock)), step=1)
                day_index = st.number_input("Day index (1–30)", min_value=1, max_value=30, value=15, step=1)
            with c3:
                weather_code = st.selectbox("Weather code", options=[0, 1], help="The meaning of codes 0/1 is not documented in the legacy dataset source.")
                special_day = st.checkbox("Special day", value=False, help="Use according to your data definition.")
            submitted = st.form_submit_button("Analyze Inventory Risk", type="primary", use_container_width=True)

    if submitted:
        if not model_exists:
            st.error("Inventory model not found. Run `python src/train_model.py` first.")
        else:
            try:
                result = predict_food_waste(int(stock), int(sold), int(day_index), int(weather_code), bool(special_day))
                st.session_state.inventory_result = {**result, "product_name": product_name.strip() or "Unnamed product"}
            except (ValueError, TypeError, OSError) as exc:
                st.session_state.inventory_result = None
                st.error(str(exc))

    inventory_result = st.session_state.get("inventory_result")
    if inventory_result:
        st.markdown('<div class="section-heading">Inventory Analysis Results</div>', unsafe_allow_html=True)
        risky = inventory_result["prediction"] == 1
        result_col, remaining_col, percentage_col = st.columns([1.4, 1, 1])
        with result_col:
            if risky:
                st.warning(f"{inventory_result['product_name']}: potential waste risk")
            else:
                st.success(f"{inventory_result['product_name']}: lower risk")
        remaining_col.metric("Remaining stock", f"{inventory_result['remaining']}")
        percentage_col.metric("Remaining inventory", f"{inventory_result['remaining_percentage']:.1f}%")
        st.write(inventory_result["recommendation"])
        st.caption("This model uses 80 example rows. Its score is not a calibrated probability and does not guarantee real-world performance.")

    with st.expander("Analyze multiple products with CSV (optional)"):
        st.write("Batch analysis remains available as an optional tool; photo screening is still SIGAP’s primary feature.")
        template = pd.DataFrame([
            {"product_name": "Apple", "stock": 100, "sold": 65, "day_index": 15, "weather_code": 0, "special_day": 0},
            {"product_name": "Banana", "stock": 80, "sold": 30, "day_index": 8, "weather_code": 1, "special_day": 0},
        ])
        d1, d2 = st.columns([1, 1])
        with d1:
            st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8-sig"), "sigap_inventory_template.csv", "text/csv", use_container_width=True)
        with d2:
            if not sample.empty:
                st.download_button("Download legacy sample data", prepare_inventory_csv_for_download(sample).to_csv(index=False).encode("utf-8-sig"), "sales_dataset.csv", "text/csv", use_container_width=True)
        batch_file = st.file_uploader("Choose inventory CSV", type=["csv"], key="inventory_csv_upload")
        if batch_file is not None:
            if not model_exists:
                st.error("Inventory model is unavailable. Run `python src/train_model.py` first.")
            else:
                try:
                    incoming = pd.read_csv(batch_file)
                    with st.spinner("Processing CSV rows…"):
                        batch_results = predict_batch(incoming)
                    display_results = prepare_batch_results_for_display(batch_results)
                    valid_count = int((batch_results["validation_status"] == "Valid").sum())
                    invalid_count = int((batch_results["validation_status"] == "Invalid").sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Rows", len(batch_results))
                    m2.metric("Valid", valid_count)
                    m3.metric("Needs Review", invalid_count)
                    st.dataframe(display_results, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download analysis results",
                        display_results.to_csv(index=False).encode("utf-8-sig"),
                        "sigap_analysis_results.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(f"Could not process CSV: {exc}")


def render_model_page() -> None:
    st.markdown('<h1 class="page-title">Models & Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Two models for two different tasks.</p>', unsafe_allow_html=True)

    photo_exists = DEFAULT_MODEL_PATH.is_file()
    inventory_exists = INVENTORY_MODEL_PATH.is_file()
    data = load_inventory_sample()
    metadata = read_json(INVENTORY_METADATA_PATH)
    photo_metrics = read_json(PHOTO_METRICS_PATH)

    photo_col, inventory_col = st.columns(2, gap="large")
    with photo_col:
        with st.container(border=True):
            st.markdown("#### 1. Photo Model")
            st.markdown("**MobileNetV3-Small · PyTorch**")
            if photo_exists:
                st.success("Photo checkpoint available")
            else:
                st.warning("Checkpoint not available")
                st.code("python scripts/download_dataset.py\npython scripts/train_model.py", language="powershell")
            if photo_metrics:
                metrics = photo_metrics.get("test_metrics", {})
                if metrics:
                    m1, m2 = st.columns(2)
                    m1.metric("Test accuracy", f"{float(metrics.get('accuracy', 0)):.1%}")
                    m2.metric("Test macro-F1", f"{float(metrics.get('macro_f1', 0)):.1%}")
                    st.caption("Scores apply to this project’s test split and are not a guarantee for all photos.")
            elif photo_exists:
                st.caption("Training metrics are not available.")
            else:
                st.caption("The photo model is trained on Fresh and Rotten image classes.")

    with inventory_col:
        with st.container(border=True):
            st.markdown("#### 2. Legacy Inventory Model")
            st.markdown(f"**{metadata.get('model', 'Decision Tree')} · scikit-learn**")
            if inventory_exists:
                st.success("Inventory model available")
            else:
                st.warning("Inventory model file not found")
            if metadata.get("holdout_metrics"):
                mm = metadata["holdout_metrics"]
                m1, m2 = st.columns(2)
                m1.metric("Holdout accuracy", f"{float(mm.get('accuracy', 0)):.1%}")
                m2.metric("Holdout macro-F1", f"{float(mm.get('f1_macro', 0)):.1%}")
            st.caption("The dataset contains only 80 rows. High scores may occur because labels are derived from leftover-stock rules.")

    if metadata.get("data_quality", {}).get("label_separation_warning"):
        st.warning("Data quality note: Safe and Potential Waste labels are clearly separated by the remaining-stock percentage. This evaluation does not prove predictive performance on real inventory.")

    if not data.empty:
        st.markdown('<div class="section-heading">Legacy Inventory Dataset</div>', unsafe_allow_html=True)
        a, b, c = st.columns([1, 1, 1.2])
        a.metric("Rows", len(data))
        b.metric("Columns", len(data.columns))
        c.download_button("Download sales dataset CSV", prepare_inventory_csv_for_download(data).to_csv(index=False).encode("utf-8-sig"), "sales_dataset.csv", "text/csv", use_container_width=True)
        st.dataframe(prepare_inventory_data_for_display(data.head(12)), use_container_width=True, hide_index=True)
        st.caption("Labels: 0 = Safe, 1 = Potential Waste. The weather-code meaning is undocumented in the source data.")

    if TREE_IMAGE_PATH.is_file():
        with st.expander("View Decision Tree structure"):
            st.image(str(TREE_IMAGE_PATH), use_container_width=True)

    if not photo_exists:
        st.info("Photo screening is the primary feature. The legacy inventory dataset and model are included and can be used without training the photo model first.")


render_header()
nav = st.radio(
    "Navigation",
    options=["Photo Check", "Inventory Analysis", "Models & Data"],
    horizontal=True,
    label_visibility="collapsed",
    key="sigap_navigation",
)
st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

if nav == "Photo Check":
    render_photo_page()
elif nav == "Inventory Analysis":
    render_inventory_page()
else:
    render_model_page()

st.markdown(
    '<div class="footer-note"><span class="footer-mark">S</span><span>SIGAP · Local AI · No API</span></div>',
    unsafe_allow_html=True,
)
