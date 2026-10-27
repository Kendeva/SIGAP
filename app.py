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
    page_title="SIGAP · Analisis Pangan",
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
          <div class="local-pill"><span class="status-dot"></span> AI lokal · Tanpa API</div>
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
    st.markdown('<h1 class="page-title">Cek kesegaran buah</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Unggah foto untuk memeriksa indikasi pembusukan.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih foto buah",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="JPG, PNG, atau WEBP · Maksimal 8 MB.",
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
            st.error("Ukuran foto maksimal 8 MB.")
        else:
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    width, height = image.size
                    if width * height > MAX_IMAGE_PIXELS:
                        st.error("Resolusi foto terlalu besar. Pilih foto yang lebih kecil.")
                    else:
                        image_dimensions = (width, height)
                        image_valid = True
            except (UnidentifiedImageError, OSError):
                st.error("Foto tidak dapat dibaca. Coba file JPG, PNG, atau WEBP lain.")

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
                st.caption("Foto terpilih")
            with meta_right:
                if image_dimensions:
                    st.caption(f"{image_dimensions[0]} × {image_dimensions[1]} px", help="Resolusi foto")
        else:
            st.markdown(
                """
                <div class="empty-preview">
                  <div class="empty-icon">↥</div>
                  <div class="empty-title">Pratinjau foto</div>
                  <div class="empty-description">Pilih satu foto buah untuk memulai.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="panel-kicker">PEMERIKSAAN FOTO</div>', unsafe_allow_html=True)
        if image_valid:
            st.markdown('<div class="panel-title">Foto siap dianalisis</div>', unsafe_allow_html=True)
            if uploaded:
                st.caption(uploaded.name)
        else:
            st.markdown('<div class="panel-title">Mulai dari foto</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-copy">Pastikan buah terlihat jelas dan pencahayaan cukup.</div>', unsafe_allow_html=True)

        if photo_model_exists:
            st.markdown('<div class="model-status"><span class="status-dot"></span> Model foto tersedia</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="model-status model-missing"><span class="status-dot"></span> Model foto belum tersedia</div>', unsafe_allow_html=True)

        has_result = st.session_state.photo_result is not None
        can_analyze = bool(image_valid and image_bytes and photo_model_exists)
        if st.button("Analisis ulang" if has_result else "Analisis foto", type="primary", use_container_width=True, disabled=not can_analyze, key="analyze_photo_button"):
            try:
                with st.spinner("Menganalisis foto secara lokal…"):
                    st.session_state.photo_result = predict_photo(image_bytes, get_cached_photo_model())
            except Exception as exc:
                st.session_state.photo_result = None
                st.error("Foto gagal dianalisis. Periksa checkpoint model SIGAP.")
                with st.expander("Detail teknis"):
                    st.code(f"{type(exc).__name__}: {exc}")

        if not photo_model_exists:
            with st.expander("Siapkan model foto"):
                st.write("Dataset foto perlu tersedia dan model perlu dilatih satu kali.")
                st.code("python scripts/download_dataset.py\npython scripts/train_model.py", language="powershell")

    result = st.session_state.photo_result
    if result:
        kind = result.get("assessment", {}).get("kind", "cannot_assess")
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        if kind == "possible_spoilage":
            title, result_class, icon = "Indikasi pembusukan", "result-warning", "!"
            advice = "Periksa buah secara langsung. Jika ada jamur atau kerusakan jelas, lebih aman untuk tidak mengonsumsinya. Jangan mencicipi untuk menguji."
        elif kind == "fresh_prediction":
            title, result_class, icon = "Terindikasi segar", "result-success", "✓"
            advice = "Foto tidak menunjukkan indikasi kuat menurut model. Tetap periksa bau, tekstur, dan cara penyimpanan sebelum dikonsumsi."
        else:
            title, result_class, icon = "Hasil belum dapat ditentukan", "result-neutral", "?"
            advice = "Coba foto ulang dengan buah terlihat utuh, fokus, dan pencahayaan cukup."

        st.markdown('<div class="section-heading">Hasil pemeriksaan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            result_col, score_col = st.columns([1.6, 0.7], gap="large")
            with result_col:
                st.markdown(
                    f'<div class="result-heading {result_class}"><span class="result-icon">{icon}</span><div><div class="result-eyebrow">HASIL ANALISIS</div><div class="result-title">{title}</div></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="result-description">{advice}</div>', unsafe_allow_html=True)
            with score_col:
                st.markdown('<div class="score-label">Skor model</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-number">{score:.0%}</div>', unsafe_allow_html=True)
                st.progress(score)
                st.caption("Bukan probabilitas keamanan pangan")

        st.caption("SIGAP hanya menilai tampilan foto. Hasil ini bukan jaminan makanan aman dikonsumsi.")


def localize_batch_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    if "status" in display.columns:
        display["status"] = display["status"].map({
            "Potential Waste": "Potensi terbuang",
            "Safe": "Risiko rendah",
            "Not evaluated": "Tidak dinilai",
        }).fillna(display["status"])
    if "risk_level" in display.columns:
        display["risk_level"] = display["risk_level"].map({
            "HIGH RISK": "Risiko tinggi",
            "LOW RISK": "Risiko rendah",
            "—": "—",
        }).fillna(display["risk_level"])
    display = display.rename(columns={
        "product_name": "Nama produk", "source_row": "Baris CSV", "status": "Status",
        "risk_level": "Tingkat risiko", "stok": "Stok", "terjual": "Terjual",
        "sisa_stok": "Sisa stok", "sisa_persen": "Sisa (%)",
        "model_confidence": "Skor model", "recommendation": "Saran",
        "validation_status": "Validasi", "validation_error": "Pesan validasi",
    })
    return display


def render_inventory_page() -> None:
    st.markdown('<h1 class="page-title">Analisis stok</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Gunakan model Decision Tree lama untuk menilai risiko sisa persediaan.</p>', unsafe_allow_html=True)

    sample = load_inventory_sample()
    model_exists = INVENTORY_MODEL_PATH.is_file()
    if not sample.empty:
        total = int(len(sample))
        potential = int((pd.to_numeric(sample.get("label"), errors="coerce") == 1).sum())
        safe = int((pd.to_numeric(sample.get("label"), errors="coerce") == 0).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Data sampel", f"{total}", help="Data contoh yang disertakan dalam proyek.")
        k2.metric("Potensi terbuang", f"{potential}")
        k3.metric("Risiko rendah", f"{safe}")
    else:
        st.info("Dataset contoh tidak ditemukan. Kamu tetap bisa menggunakan input manual jika model tersedia.")

    with st.container(border=True):
        st.markdown("#### Periksa satu produk")
        with st.form("single_inventory_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                product_name = st.text_input("Nama produk", value="Produk contoh", max_chars=80)
                stock = st.number_input("Jumlah stok", min_value=1, max_value=100000, value=100, step=1)
            with c2:
                sold = st.number_input("Jumlah terjual", min_value=0, max_value=int(stock), value=min(65, int(stock)), step=1)
                day_index = st.number_input("Hari ke (1–30)", min_value=1, max_value=30, value=15, step=1)
            with c3:
                weather_code = st.selectbox("Kode cuaca", options=[0, 1], help="Arti kode 0/1 tidak dijelaskan dalam sumber dataset lama.")
                special_day = st.checkbox("Hari khusus", value=False, help="Gunakan sesuai definisi data milikmu.")
            submitted = st.form_submit_button("Analisis risiko stok", type="primary", use_container_width=True)

    if submitted:
        if not model_exists:
            st.error("Model stok belum ditemukan. Jalankan `python src/train_model.py` terlebih dahulu.")
        else:
            try:
                result = predict_food_waste(int(stock), int(sold), int(day_index), int(weather_code), bool(special_day))
                st.session_state.inventory_result = {**result, "product_name": product_name.strip() or "Produk tanpa nama"}
            except (ValueError, TypeError, OSError) as exc:
                st.session_state.inventory_result = None
                st.error(str(exc))

    inventory_result = st.session_state.get("inventory_result")
    if inventory_result:
        st.markdown('<div class="section-heading">Hasil analisis stok</div>', unsafe_allow_html=True)
        risky = inventory_result["prediction"] == 1
        result_col, remaining_col, percentage_col = st.columns([1.4, 1, 1])
        with result_col:
            if risky:
                st.warning(f"{inventory_result['product_name']}: potensi terbuang")
            else:
                st.success(f"{inventory_result['product_name']}: risiko rendah")
        remaining_col.metric("Sisa stok", f"{inventory_result['remaining']}")
        percentage_col.metric("Sisa persediaan", f"{inventory_result['remaining_percentage']:.1f}%")
        st.write(inventory_result["recommendation"])
        st.caption("Model ini berdasarkan 80 baris data contoh. Skor model bukan probabilitas terkalibrasi dan tidak menjamin hasil di kondisi nyata.")

    with st.expander("Analisis banyak produk melalui CSV (opsional)"):
        st.write("Fitur lama tetap tersedia untuk analisis batch; pemeriksaan foto tetap menjadi fitur utama SIGAP.")
        template = pd.DataFrame([
            {"product_name": "Apel", "stok": 100, "terjual": 65, "hari_ke": 15, "cuaca": 0, "hari_besar": 0},
            {"product_name": "Pisang", "stok": 80, "terjual": 30, "hari_ke": 8, "cuaca": 1, "hari_besar": 0},
        ])
        d1, d2 = st.columns([1, 1])
        with d1:
            st.download_button("Unduh template CSV", template.to_csv(index=False).encode("utf-8-sig"), "sigap_template_stok.csv", "text/csv", use_container_width=True)
        with d2:
            if not sample.empty:
                st.download_button("Unduh data sampel lama", sample.to_csv(index=False).encode("utf-8-sig"), "data_penjualan.csv", "text/csv", use_container_width=True)
        batch_file = st.file_uploader("Pilih CSV stok", type=["csv"], key="inventory_csv_upload")
        if batch_file is not None:
            if not model_exists:
                st.error("Model stok belum tersedia. Jalankan `python src/train_model.py` terlebih dahulu.")
            else:
                try:
                    incoming = pd.read_csv(batch_file)
                    with st.spinner("Memproses baris CSV…"):
                        batch_results = predict_batch(incoming)
                    display_results = localize_batch_results(batch_results)
                    valid_count = int((batch_results["validation_status"] == "Valid").sum())
                    invalid_count = int((batch_results["validation_status"] == "Invalid").sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total baris", len(batch_results))
                    m2.metric("Valid", valid_count)
                    m3.metric("Perlu diperbaiki", invalid_count)
                    st.dataframe(display_results, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Unduh hasil analisis",
                        display_results.to_csv(index=False).encode("utf-8-sig"),
                        "sigap_hasil_analisis.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                    st.error(f"CSV tidak dapat diproses: {exc}")


def render_model_page() -> None:
    st.markdown('<h1 class="page-title">Model & data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Dua model berbeda untuk dua tugas yang berbeda.</p>', unsafe_allow_html=True)

    photo_exists = DEFAULT_MODEL_PATH.is_file()
    inventory_exists = INVENTORY_MODEL_PATH.is_file()
    data = load_inventory_sample()
    metadata = read_json(INVENTORY_METADATA_PATH)
    photo_metrics = read_json(PHOTO_METRICS_PATH)

    photo_col, inventory_col = st.columns(2, gap="large")
    with photo_col:
        with st.container(border=True):
            st.markdown("#### 1. Model foto")
            st.markdown("**MobileNetV3-Small · PyTorch**")
            if photo_exists:
                st.success("Checkpoint foto tersedia")
            else:
                st.warning("Checkpoint belum tersedia")
                st.code("python scripts/download_dataset.py\npython scripts/train_model.py", language="powershell")
            if photo_metrics:
                metrics = photo_metrics.get("test_metrics", {})
                if metrics:
                    m1, m2 = st.columns(2)
                    m1.metric("Test accuracy", f"{float(metrics.get('accuracy', 0)):.1%}")
                    m2.metric("Test macro-F1", f"{float(metrics.get('macro_f1', 0)):.1%}")
                    st.caption("Skor berlaku pada test split proyek, bukan jaminan untuk semua foto.")
            elif photo_exists:
                st.caption("Metadata metrik training belum tersedia.")
            else:
                st.caption("Model foto dibuat dari dataset gambar Fresh dan Rotten.")

    with inventory_col:
        with st.container(border=True):
            st.markdown("#### 2. Model stok lama")
            st.markdown(f"**{metadata.get('model', 'Decision Tree')} · scikit-learn**")
            if inventory_exists:
                st.success("Model stok tersedia")
            else:
                st.warning("File model stok tidak ditemukan")
            if metadata.get("holdout_metrics"):
                mm = metadata["holdout_metrics"]
                m1, m2 = st.columns(2)
                m1.metric("Holdout accuracy", f"{float(mm.get('accuracy', 0)):.1%}")
                m2.metric("Holdout macro-F1", f"{float(mm.get('f1_macro', 0)):.1%}")
            st.caption("Dataset hanya 80 baris; skor tinggi bisa muncul karena label dibuat dari aturan sisa stok.")

    if metadata.get("data_quality", {}).get("label_separation_warning"):
        st.warning("Catatan kualitas data: label Safe dan Potential Waste terpisah jelas oleh persentase sisa stok. Evaluasi ini belum membuktikan kemampuan prediksi pada stok nyata.")

    if not data.empty:
        st.markdown('<div class="section-heading">Dataset stok lama</div>', unsafe_allow_html=True)
        a, b, c = st.columns([1, 1, 1.2])
        a.metric("Jumlah baris", len(data))
        b.metric("Kolom", len(data.columns))
        c.download_button("Unduh data_penjualan.csv", data.to_csv(index=False).encode("utf-8-sig"), "data_penjualan.csv", "text/csv", use_container_width=True)
        st.dataframe(data.head(12), use_container_width=True, hide_index=True)
        st.caption("Label: 0 = Safe, 1 = Potential Waste. Makna kode cuaca pada data sumber tidak terdokumentasi.")

    if TREE_IMAGE_PATH.is_file():
        with st.expander("Lihat struktur Decision Tree"):
            st.image(str(TREE_IMAGE_PATH), use_container_width=True)

    if not photo_exists:
        st.info("Foto menjadi fitur utama. Dataset dan model stok lama tetap disertakan dan dapat digunakan tanpa melatih model foto terlebih dahulu.")


render_header()
nav = st.radio(
    "Navigasi",
    options=["Cek Foto", "Analisis Stok", "Model & Data"],
    horizontal=True,
    label_visibility="collapsed",
    key="sigap_navigation",
)
st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

if nav == "Cek Foto":
    render_photo_page()
elif nav == "Analisis Stok":
    render_inventory_page()
else:
    render_model_page()

st.markdown(
    '<div class="footer-note"><span class="footer-mark">S</span><span>SIGAP · Analisis lokal · Tanpa API</span></div>',
    unsafe_allow_html=True,
)
