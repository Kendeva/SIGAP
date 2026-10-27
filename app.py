"""SIGAP: simple one-page local fruit freshness checker."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.local_vision import DEFAULT_MODEL_PATH, load_local_model, predict_photo

BASE_DIR = Path(__file__).resolve().parent
STYLE_PATH = BASE_DIR / "assets" / "styles.css"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="SIGAP · Cek Kesegaran",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if STYLE_PATH.exists():
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="brand"><span class="brand-icon">♻</span><span>SIGAP</span></div>
    <h1 class="page-title">Cek kesegaran buah</h1>
    <p class="page-subtitle">Unggah foto untuk memeriksa indikasi pembusukan.</p>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Pilih foto buah",
    type=SUPPORTED_TYPES,
    accept_multiple_files=False,
    help="JPG, PNG, atau WEBP. Maksimal 8 MB.",
)

image_bytes: bytes | None = None
image_valid = False
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.error("Ukuran foto maksimal 8 MB.")
    else:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    st.error("Resolusi foto terlalu besar. Pilih foto dengan ukuran lebih kecil.")
                else:
                    st.image(image.convert("RGB"), use_container_width=True)
                    image_valid = True
        except (UnidentifiedImageError, OSError):
            st.error("Foto tidak dapat dibaca. Coba file JPG, PNG, atau WEBP lain.")

@st.cache_resource(show_spinner=False)
def get_cached_model():
    return load_local_model()

model_exists = DEFAULT_MODEL_PATH.is_file()
can_analyze = bool(uploaded is not None and image_bytes is not None and image_valid and model_exists)
if st.button("Analisis foto", type="primary", use_container_width=True, disabled=not can_analyze):
    try:
        with st.spinner("Menganalisis foto…"):
            result = predict_photo(image_bytes, get_cached_model())
        kind = result["assessment"]["kind"]
        if kind == "possible_spoilage":
            st.warning("**Indikasi pembusukan**")
        elif kind == "fresh_prediction":
            st.success("**Terindikasi segar**")
        else:
            st.info("**Hasil belum dapat ditentukan**")
    except Exception as exc:
        st.error("Foto gagal dianalisis. Periksa instalasi dan model SIGAP.")
        with st.expander("Detail teknis"):
            st.code(f"{type(exc).__name__}: {exc}")

if not model_exists:
    with st.expander("Model belum tersedia — cara menyiapkan"):
        st.write("Unduh dataset dan latih model terlebih dahulu:")
        st.code("python scripts/download_dataset.py\npython scripts/train_model.py", language="powershell")

st.caption("Hasil foto hanya pemeriksaan awal dan bukan jaminan keamanan pangan.")
