"""
CabaiDetect — Deteksi Buah Cabai Berbasis CNN EfficientNetB0
Jalankan dengan: streamlit run app.py
"""

import io
import base64
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="CabaiDetect — Deteksi Buah Cabai AI",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #d4f5e9 0%, #b2edd8 30%, #e8faf3 60%, #f0fff8 100%) !important;
    min-height: 100vh;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stMain"] > div { padding-top: 0 !important; }
[data-testid="block-container"] {
    padding: 0 1.5rem 3rem 1.5rem !important;
    max-width: 1100px; margin: 0 auto;
}

/* NAVBAR */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 2rem;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(255,255,255,0.7);
    position: sticky; top: 0; z-index: 999;
    margin: 0 -1.5rem 0 -1.5rem;
}
.navbar-brand { display: flex; align-items: center; gap: 0.55rem; }
.navbar-logo {
    width: 36px; height: 36px; background: #00c896;
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 18px;
}
.navbar-name { font-weight: 800; font-size: 1.15rem; color: #0d1f17; }
.navbar-name span { color: #00c896; }
.navbar-links { display: flex; gap: 2rem; }
.navbar-links a { text-decoration: none; color: #2d4a3e; font-weight: 500; font-size: 0.9rem; }
.btn-nav {
    background: #00c896; color: #fff !important;
    padding: 0.55rem 1.3rem; border-radius: 50px;
    font-weight: 700; font-size: 0.88rem;
    text-decoration: none; display: inline-flex;
    align-items: center; gap: 0.4rem;
    box-shadow: 0 4px 15px rgba(0,200,150,0.35);
}

/* HERO */
.hero { text-align: center; padding: 4.5rem 1rem 3rem; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(0,200,150,0.25);
    padding: 0.4rem 1.1rem; border-radius: 50px;
    font-size: 0.82rem; font-weight: 600; color: #008f6a;
    margin-bottom: 1.6rem;
}
.hero-badge-dot { width:7px; height:7px; background:#00c896; border-radius:50%; }
.hero h1 { font-size: clamp(2.2rem, 5vw, 3.2rem); font-weight: 800; color: #0d1f17; line-height: 1.2; }
.hero h1 .accent { color: #00c896; }
.hero-sub { color: #4a7060; font-size: 1rem; line-height: 1.7; max-width: 560px; margin: 1.1rem auto 2rem; text-align: center !important; display: block; width: 100%; }

/* Sembunyikan anchor icon otomatis Streamlit pada heading */
.hero h1 a, .hero h1 .anchor-link { display: none !important; }
/* Override Streamlit default text-align pada elemen dalam hero */
.hero p, .hero div { text-align: center !important; }
[data-testid="stMarkdownContainer"] .hero p { text-align: center !important; }
.hero-btns { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
.btn-primary {
    background: #00c896; color: #fff; padding: 0.75rem 1.7rem; border-radius: 50px;
    font-weight: 700; font-size: 0.93rem; border: none;
    display: inline-flex; align-items: center; gap: 0.45rem;
    box-shadow: 0 6px 20px rgba(0,200,150,0.4); text-decoration: none;
}
.btn-secondary {
    background: #fff; color: #0d1f17; padding: 0.75rem 1.7rem; border-radius: 50px;
    font-weight: 700; font-size: 0.93rem;
    border: 1.5px solid rgba(0,200,150,0.3);
    display: inline-flex; align-items: center; gap: 0.45rem; text-decoration: none;
}

/* STATS */
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin: 0.5rem 0 3.5rem; }
.stat-card {
    background: rgba(255,255,255,0.75); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.9); border-radius: 20px;
    padding: 1.6rem 1.4rem; text-align: center;
    box-shadow: 0 4px 24px rgba(0,60,40,0.07);
}
.stat-value { font-size: 2.2rem; font-weight: 800; color: #00c896; }
.stat-label { font-size: 0.82rem; color: #5a7a6a; font-weight: 500; }
.stat-bar { height: 4px; background: #00c896; border-radius: 4px; margin: 0.8rem auto 0; width: 50%; }
.stat-dots { font-size: 1.2rem; color: #00c896; margin-top: 0.5rem; }
.stat-check { font-size: 1.5rem; color: #00c896; margin-top: 0.5rem; }

/* SECTION TITLE */
.section-title { text-align: center; margin-bottom: 2rem; }
.section-title h2 { font-size: clamp(1.6rem, 4vw, 2.2rem); font-weight: 800; color: #0d1f17; }
.section-title h2 .accent { color: #00c896; }
.section-title p { color: #5a7a6a; margin-top: 0.5rem; font-size: 0.93rem; }

/* STEPS */
.steps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin-bottom: 4rem; }
.step-card {
    background: rgba(255,255,255,0.65); border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px; padding: 1.5rem 1.2rem; text-align: center;
    box-shadow: 0 3px 18px rgba(0,60,40,0.06);
}
.step-num {
    width: 42px; height: 42px; background: #00c896; border-radius: 50%;
    color: #fff; font-weight: 800; font-size: 1.1rem;
    display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;
}
.step-title { font-weight: 700; color: #0d1f17; margin-bottom: 0.4rem; }
.step-desc { color: #5a7a6a; font-size: 0.83rem; line-height: 1.6; }

/* DROP ZONE — sebelum ada gambar */
[data-testid="stFileUploader"] { max-width: 720px; margin: 0 auto; }
[data-testid="stFileUploader"] label { display: none !important; }

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(0,200,150,0.6) !important;
    border-radius: 18px !important;
    background: rgba(240,255,250,0.6) !important;
    min-height: 240px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 2.5rem 1.5rem !important;
    transition: all 0.22s ease !important;
    cursor: pointer !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00c896 !important;
    background: rgba(0,200,150,0.08) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::before {
    content: "🖼️";
    display: flex; align-items: center; justify-content: center;
    width: 64px; height: 64px; background: #00c896;
    border-radius: 50%; font-size: 26px; margin-bottom: 0.8rem;
    box-shadow: 0 4px 16px rgba(0,200,150,0.4);
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important; color: #0d1f17 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important; color: #7a9a8a !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important; color: #00c896 !important;
    border: 1.5px solid rgba(0,200,150,0.5) !important;
    border-radius: 50px !important; font-weight: 700 !important;
    font-size: 0.82rem !important; padding: 0.4rem 1.2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; margin-top: 0.5rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(0,200,150,0.1) !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #7a9a8a !important; font-size: 0.78rem !important;
}

/* TOMBOL ANALISIS — hijau besar */
div[data-testid="stButton"].btn-analisis > button {
    background: linear-gradient(90deg, #00c896, #00b386) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.9rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(0,200,150,0.35) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
/* Fallback: semua button hijau kecuali yang punya class khusus */
[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #00c896, #00b386) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.9rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(0,200,150,0.35) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: opacity 0.18s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.9 !important; }

/* TOMBOL GANTI GAMBAR — override khusus via class wrapper */
.ganti-btn [data-testid="stButton"] > button {
    background: rgba(255,255,255,0.9) !important;
    color: #00a07a !important;
    border: 1.5px solid rgba(0,200,150,0.55) !important;
    border-radius: 50px !important;
    font-size: 0.82rem !important;
    padding: 0.42rem 1.1rem !important;
    box-shadow: none !important;
    font-weight: 600 !important;
    width: auto !important;
}
.ganti-btn [data-testid="stButton"] > button:hover {
    background: rgba(0,200,150,0.08) !important;
    opacity: 1 !important;
}

/* CHECKBOX */
[data-testid="stCheckbox"] { max-width: 720px; margin: 0 auto 0.9rem; }
[data-testid="stCheckbox"] label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important; color: #2d5a45 !important; font-size: 0.88rem !important;
}

/* HASIL DETEKSI */
.result-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(255,255,255,0.95);
    border-radius: 24px; padding: 2rem 2.2rem;
    box-shadow: 0 6px 32px rgba(0,60,40,0.08);
    max-width: 720px; margin: 0 auto;
}
.badge-healthy {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(0,200,150,0.12); border: 1.5px solid rgba(0,200,150,0.4);
    color: #007a58; padding: 0.45rem 1.4rem;
    border-radius: 50px; font-weight: 700; font-size: 1rem; margin-bottom: 1rem;
}
.badge-unhealthy {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(244,67,54,0.1); border: 1.5px solid rgba(244,67,54,0.35);
    color: #c0392b; padding: 0.45rem 1.4rem;
    border-radius: 50px; font-weight: 700; font-size: 1rem; margin-bottom: 1rem;
}
.conf-bar-wrap {
    background: #e8f5f0; border-radius: 50px;
    height: 10px; margin: 1rem 0 0.4rem; overflow: hidden;
}
.conf-bar-green { height:100%; border-radius:50px; background:linear-gradient(90deg,#00c896,#00e6ae); }
.conf-bar-red   { height:100%; border-radius:50px; background:linear-gradient(90deg,#f44336,#ff7043); }
.metrics-row { display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin-top:1.2rem; }
.metric-box {
    background: rgba(0,200,150,0.07); border: 1px solid rgba(0,200,150,0.15);
    border-radius: 14px; padding: 0.9rem; text-align: center;
}
.metric-val { font-size:1.1rem; font-weight:800; color:#00c896; }
.metric-lbl { font-size:0.72rem; color:#5a7a6a; margin-top:0.2rem; }

/* MODEL STATUS */
.model-ok {
    background: rgba(0,200,150,0.08); border: 1px solid rgba(0,200,150,0.25);
    border-radius: 14px; padding: 0.75rem 1.3rem; font-size: 0.82rem;
    color: #2d5a45; max-width: 720px; margin: 0 auto 1.5rem; text-align: center;
}
.model-warn {
    background: rgba(255,165,0,0.07); border: 1px solid rgba(255,165,0,0.4);
    border-radius: 14px; padding: 0.75rem 1.3rem; font-size: 0.82rem;
    color: #7a5000; max-width: 720px; margin: 0 auto 1.5rem; text-align: center;
}

/* FOOTER */
.footer {
    text-align: center; padding: 2rem; color: #7a9a8a; font-size: 0.8rem;
    border-top: 1px solid rgba(0,200,150,0.15); margin-top: 3rem;
}
.divider { height:1px; background:rgba(0,200,150,0.15); margin:2.5rem 0; }


/* ══════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE — hanya berlaku di layar ≤ 768px
   Tidak ada perubahan pada tampilan desktop sama sekali.
   ══════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {

    /* Block container — kurangi padding samping */
    [data-testid="block-container"] {
        padding: 0 0.75rem 2rem 0.75rem !important;
    }

    /* ── NAVBAR ──────────────────────────────────────────────── */
    .navbar {
        padding: 0.85rem 1rem !important;
        margin: 0 -0.75rem !important;
        flex-wrap: nowrap;
        gap: 0;
    }
    /* Sembunyikan link navigasi & tombol CTA di mobile, cukup brand */
    .navbar-links { display: none !important; }
    .btn-nav { display: none !important; }
    .navbar-name { font-size: 1rem !important; }
    .navbar-logo { width: 30px !important; height: 30px !important; font-size: 15px !important; border-radius: 8px !important; }

    /* ── HERO ────────────────────────────────────────────────── */
    .hero {
        padding: 2.5rem 0.5rem 2rem !important;
    }
    .hero h1 {
        font-size: 1.85rem !important;
        line-height: 1.25 !important;
    }
    .hero-badge {
        font-size: 0.72rem !important;
        padding: 0.35rem 0.85rem !important;
        text-align: center;
        white-space: normal;
        line-height: 1.4;
    }
    .hero-sub {
        font-size: 0.88rem !important;
        margin: 0.9rem auto 1.5rem !important;
        padding: 0 0.5rem;
    }
    .hero-btns {
        flex-direction: column !important;
        align-items: center !important;
        gap: 0.7rem !important;
    }
    .btn-primary, .btn-secondary {
        width: 100% !important;
        max-width: 280px !important;
        justify-content: center !important;
        font-size: 0.88rem !important;
        padding: 0.7rem 1.2rem !important;
    }

    /* ── STATS GRID — 3 kolom → 1 baris scroll horizontal ───── */
    .stats-grid {
        grid-template-columns: repeat(3, minmax(90px, 1fr)) !important;
        gap: 0.65rem !important;
        margin: 0 0 2.5rem !important;
    }
    .stat-card {
        padding: 1.1rem 0.6rem !important;
        border-radius: 16px !important;
    }
    .stat-value { font-size: 1.6rem !important; }
    .stat-label { font-size: 0.72rem !important; }

    /* ── SECTION TITLE ───────────────────────────────────────── */
    .section-title { margin-bottom: 1.3rem !important; }
    .section-title h2 { font-size: 1.45rem !important; }
    .section-title p { font-size: 0.83rem !important; }

    /* ── STEPS GRID — 3 kolom → 1 kolom vertikal ────────────── */
    .steps-grid {
        grid-template-columns: 1fr !important;
        gap: 0.9rem !important;
        margin-bottom: 2.5rem !important;
    }
    .step-card {
        padding: 1.2rem 1rem !important;
        text-align: left !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 1rem !important;
    }
    .step-num {
        margin: 0 !important;
        flex-shrink: 0 !important;
        width: 36px !important;
        height: 36px !important;
        font-size: 0.95rem !important;
    }
    .step-content { flex: 1; }
    .step-title { margin-bottom: 0.25rem !important; font-size: 0.93rem !important; }
    .step-desc { font-size: 0.8rem !important; }

    /* ── DROP ZONE ───────────────────────────────────────────── */
    [data-testid="stFileUploader"] { max-width: 100% !important; }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 190px !important;
        padding: 1.8rem 1rem !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        width: 52px !important;
        height: 52px !important;
        font-size: 22px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        font-size: 0.88rem !important;
    }

    /* ── TOMBOL ANALISIS ─────────────────────────────────────── */
    [data-testid="stButton"] > button {
        font-size: 0.93rem !important;
        padding: 0.85rem 1rem !important;
    }

    /* ── CHECKBOX ────────────────────────────────────────────── */
    [data-testid="stCheckbox"] {
        max-width: 100% !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stCheckbox"] label {
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
    }

    /* ── MODEL STATUS ────────────────────────────────────────── */
    .model-ok, .model-warn {
        max-width: 100% !important;
        font-size: 0.78rem !important;
        padding: 0.65rem 0.9rem !important;
        line-height: 1.6 !important;
    }

    /* ── RESULT CARD ─────────────────────────────────────────── */
    .result-card {
        max-width: 100% !important;
        padding: 1.3rem 1.1rem !important;
        border-radius: 18px !important;
    }
    .badge-healthy, .badge-unhealthy {
        font-size: 0.88rem !important;
        padding: 0.38rem 1rem !important;
    }
    .metrics-row {
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 0.5rem !important;
    }
    .metric-box { padding: 0.7rem 0.4rem !important; border-radius: 10px !important; }
    .metric-val { font-size: 0.95rem !important; }
    .metric-lbl { font-size: 0.65rem !important; }

    /* ── PERBANDINGAN GAMBAR — 2 kolom tetap, lebih compact ──── */
    /* Streamlit columns di mobile sudah stack otomatis di lebar sangat sempit,
       tapi kita pastikan gambar tidak overflow */
    [data-testid="stImage"] img,
    .result-card img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* ── KOLOM STREAMLIT (upload & result area) ──────────────── */
    /* Hapus padding samping berlebih pada kolom-kolom */
    [data-testid="column"] {
        padding-left: 0.3rem !important;
        padding-right: 0.3rem !important;
        min-width: 0 !important;
    }

    /* ── GANTI GAMBAR BUTTON ─────────────────────────────────── */
    .ganti-btn [data-testid="stButton"] > button {
        font-size: 0.78rem !important;
        padding: 0.38rem 0.9rem !important;
    }

    /* ── FOOTER ──────────────────────────────────────────────── */
    .footer {
        font-size: 0.75rem !important;
        padding: 1.5rem 0.5rem !important;
        line-height: 1.7 !important;
    }

    /* ── DIVIDER ─────────────────────────────────────────────── */
    .divider { margin: 1.8rem 0 !important; }

    /* ── SECTION SPACING ANTAR ELEMEN ───────────────────────── */
    .section-title { margin-bottom: 1.2rem !important; }
}

/* Extra kecil — hp lama / layar 360px */
@media (max-width: 400px) {
    .hero h1 { font-size: 1.55rem !important; }
    .stats-grid {
        grid-template-columns: repeat(3, 1fr) !important;
        gap: 0.5rem !important;
    }
    .stat-card { padding: 0.85rem 0.4rem !important; border-radius: 12px !important; }
    .stat-value { font-size: 1.3rem !important; }
    .stat-label { font-size: 0.65rem !important; }
    .hero-badge { font-size: 0.67rem !important; }
    .metrics-row { gap: 0.35rem !important; }
    .metric-val { font-size: 0.85rem !important; }
    .result-card { padding: 1rem 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FUNGSI BACKEND
# ══════════════════════════════════════════════════════════════
import gdown
import os

MODEL_PATH = "efficientnet_cabai_final.keras"

if not os.path.exists(MODEL_PATH):
    FILE_ID = "1VPjmD3MWSfsSTDMgdG4uWCIWYm4gS-Mo"
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def open_image_safe(file_bytes: bytes) -> Image.Image:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    img = Image.open(io.BytesIO(file_bytes))
    img.load()
    return img.convert("RGB")

def preprocess_image(pil_img: Image.Image, remove_bg: bool = True):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = pil_img.copy()
    rembg_status = None

    if remove_bg:
        try:
            from rembg import remove as rmbg, new_session
            session = new_session("u2net")
            img_no_bg = rmbg(img, session=session)
            canvas = Image.new("RGB", img_no_bg.size, (255, 255, 255))
            if img_no_bg.mode == "RGBA":
                canvas.paste(img_no_bg, mask=img_no_bg.split()[3])
            else:
                canvas.paste(img_no_bg)
            img = canvas
            rembg_status = True
        except ImportError:
            rembg_status = "install"
        except Exception as e:
            rembg_status = str(e)

    img_resized = img.resize((224, 224), Image.LANCZOS)
    arr = preprocess_input(np.array(img_resized, dtype=np.float32))
    return np.expand_dims(arr, axis=0), img_resized, rembg_status

def predict(model, arr: np.ndarray, threshold: float = 0.5):
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = "unhealthy" if prob >= threshold else "healthy"
    conf  = prob * 100 if prob >= threshold else (1 - prob) * 100
    return label, conf, prob

def img_to_b64(pil_img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════
for key, val in [("uploaded_bytes", None), ("uploaded_name", None), ("model", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
if st.session_state.model is None:
    try:
        with st.spinner("🔄 Memuat model AI..."):
            st.session_state.model = load_model(MODEL_PATH)
    except Exception:
        st.session_state.model = None


# ══════════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<nav class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">🌶️</div>
    <span class="navbar-name">Cabai<span>Detect</span></span>
  </div>
  <div class="navbar-links">
    <a href="#beranda" onclick="document.getElementById('beranda').scrollIntoView({behavior:'smooth'});return false;">Beranda</a>
    <a href="#section-cara-kerja" onclick="document.getElementById('section-cara-kerja').scrollIntoView({behavior:'smooth'});return false;">Fitur</a>
    <a href="#section-demo" onclick="document.getElementById('section-demo').scrollIntoView({behavior:'smooth'});return false;">Demo</a>
    <a href="#section-cara-kerja" onclick="document.getElementById('section-cara-kerja').scrollIntoView({behavior:'smooth'});return false;">Tentang</a>
  </div>
  <a class="btn-nav" href="#section-demo" onclick="document.getElementById('section-demo').scrollIntoView({behavior:'smooth'});return false;">📷 Mulai Deteksi →</a>
</nav>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO + STATS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div id="beranda" class="hero">
  <div class="hero-badge">
    <div class="hero-badge-dot"></div>
    Teknologi CNN EfficientNetB0 — Transfer Learning
  </div>
  <h1>Deteksi Dini<br><span class="accent">Buah Cabai</span><br>dengan Kecerdasan Buatan</h1>
  <p class="hero-sub" style="text-align:center !important; margin-left:auto; margin-right:auto; display:block; width:100%;">
    Sistem berbasis Deep Learning untuk mengklasifikasikan kondisi buah cabai
    <strong>Healthy</strong> atau <strong>Unhealthy</strong> 
  </p>
  <div class="hero-btns">
    <a class="btn-primary" href="#section-demo" onclick="document.getElementById('section-demo').scrollIntoView({behavior:'smooth'});return false;">📷 Mulai Deteksi →</a>
    <a class="btn-secondary" href="#section-cara-kerja" onclick="document.getElementById('section-cara-kerja').scrollIntoView({behavior:'smooth'});return false;">▶ Lihat Cara Kerja</a>
  </div>
</div>
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">97%+</div>
    <div class="stat-label">Akurasi Deteksi</div>
    <div class="stat-bar"></div>
  </div>
  <div class="stat-card">
    <div class="stat-value">&lt;2s</div>
    <div class="stat-label">Waktu Proses</div>
    <div class="stat-dots">● ● ●</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">2</div>
    <div class="stat-label">Kelas Deteksi</div>
    <div class="stat-check">✅</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CARA KERJA
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div id="section-cara-kerja" class="section-title">
  <h2>Cara <span class="accent">Kerja Sistem</span></h2>
  <p>Proses deteksi dilakukan secara otomatis dalam 3 langkah</p>
</div>
<div class="steps-grid">
  <div class="step-card">
    <div class="step-num">1</div>
    <div class="step-content">
      <div class="step-title">Upload Gambar</div>
      <div class="step-desc">Upload foto buah cabai format JPG, PNG, atau HEIC. Sistem langsung memproses gambar Anda.</div>
    </div>
  </div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div class="step-content">
      <div class="step-title">Preprocessing AI</div>
      <div class="step-desc">Remove background otomatis, resize ke 224×224, dan normalisasi standar EfficientNet.</div>
    </div>
  </div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div class="step-content">
      <div class="step-title">Hasil Klasifikasi</div>
      <div class="step-desc">Model CNN menghasilkan prediksi <strong>Healthy</strong> atau <strong>Unhealthy</strong> beserta skor keyakinan.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DEMO SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div id="section-demo" class="section-title">
  <h2>Coba <span class="accent">Deteksi Sekarang</span></h2>
  <p>Upload gambar cabai dan biarkan AI menganalisisnya dalam hitungan detik</p>
</div>
""", unsafe_allow_html=True)

# Status model
if st.session_state.model:
    st.markdown("""
    <div class="model-ok">
      ✅ <b>Model aktif &amp; siap digunakan</b> &nbsp;|&nbsp;
      🧠 EfficientNetB0 &nbsp;|&nbsp; 📐 Input: 224×224×3
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="model-warn">
      ⚠️ <b>Model gagal dimuat.</b> Pastikan file
      <b>efficientnet_cabai_final.keras</b> ada di folder yang sama dengan app.py
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
#  UPLOAD SECTION
# ══════════════════════════════════════════════════════════════
_, col_main, _ = st.columns([0.1, 0.8, 0.1])

with col_main:

    # Label
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;
                font-weight:700;font-size:1rem;color:#0d1f17;margin-bottom:1rem;">
      <div style="width:34px;height:34px;background:#00c896;border-radius:9px;
                  display:flex;align-items:center;justify-content:center;font-size:16px;">☁️</div>
      Upload Gambar Cabai
    </div>
    """, unsafe_allow_html=True)

    has_image = st.session_state.uploaded_bytes is not None

    # ── KONDISI 1: Belum ada gambar → tampilkan drop zone ────
    if not has_image:
        uploaded_file = st.file_uploader(
            "Upload gambar cabai",
            type=["jpg", "jpeg", "png", "heic"],
            label_visibility="collapsed",
            key="uploader_fresh",
        )
        if uploaded_file is not None:
            st.session_state.uploaded_bytes = uploaded_file.read()
            st.session_state.uploaded_name  = uploaded_file.name
            st.rerun()

    # ── KONDISI 2: Sudah ada gambar → tampilkan preview dalam kotak ──
    else:
        pil_img = None
        try:
            pil_img = open_image_safe(st.session_state.uploaded_bytes)
        except Exception as e:
            st.error(f"❌ Gagal membaca gambar: {e}")
            st.session_state.uploaded_bytes = None
            st.session_state.uploaded_name  = None
            st.rerun()

        if pil_img is not None:
            img_b64 = img_to_b64(pil_img)

            st.markdown(f"""
            <div style="
                border: 2px dashed rgba(0,200,150,0.6);
                border-radius: 18px;
                background: rgba(240,255,250,0.6);
                min-height: 240px;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin-bottom: 0.8rem;
            ">
              <div style="width:100%;font-size:0.74rem;font-weight:700;color:#5a7a6a;
                          margin-bottom:0.75rem;display:flex;align-items:center;gap:0.4rem;">
                🖼️ <span style="color:#00c896;word-break:break-all;">{st.session_state.uploaded_name}</span>
              </div>
              <img src="data:image/jpeg;base64,{img_b64}"
                   style="max-height:270px;max-width:90%;border-radius:12px;
                          object-fit:contain;box-shadow:0 4px 18px rgba(0,60,40,0.13);">
              <div style="margin-top:0.65rem;font-size:0.74rem;color:#7a9a8a;font-weight:500;">
                {pil_img.width} × {pil_img.height} px
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="ganti-btn" style="text-align:center;margin-bottom:0.5rem;">', unsafe_allow_html=True)
            col_l, col_btn, col_r = st.columns([0.38, 0.24, 0.38])
            with col_btn:
                if st.button("🔄 Ganti Gambar", key="btn_ganti", use_container_width=True):
                    st.session_state.uploaded_bytes = None
                    st.session_state.uploaded_name  = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Checkbox & tombol analisis (selalu tampil) ────────────
    pil_img_final = None
    if has_image and st.session_state.uploaded_bytes:
        try:
            pil_img_final = open_image_safe(st.session_state.uploaded_bytes)
        except Exception:
            pass

    remove_bg_on = st.checkbox("🪄 Aktifkan Remove Background (hapus latar belakang otomatis)", value=True)
    analyze_btn  = st.button("🔍  Analisis Gambar", use_container_width=True, key="btn_analisis")


# ══════════════════════════════════════════════════════════════
#  HASIL DETEKSI
# ══════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

if has_image and analyze_btn and pil_img_final is not None:
    try:
        with st.spinner("🧠 AI sedang menganalisis gambar..."):
            arr, img_processed, rembg_status = preprocess_image(pil_img_final, remove_bg=remove_bg_on)
            label, conf, raw_score = predict(st.session_state.model, arr)

        if remove_bg_on:
            if rembg_status is True:
                st.success("✅ Remove background berhasil diterapkan.")
            elif rembg_status == "install":
                st.warning("⚠️ Library `rembg` belum terinstall. Jalankan: `pip install rembg` lalu restart app.")
            elif rembg_status is not None:
                st.warning(f"⚠️ Remove background gagal, gambar asli digunakan. Detail: {rembg_status}")

        is_healthy = label == "healthy"
        color_hex  = "#00c896" if is_healthy else "#f44336"
        emoji      = "✅" if is_healthy else "⚠️"
        label_id   = "Sehat (Healthy)" if is_healthy else "Tidak Sehat (Unhealthy)"
        bar_class  = "conf-bar-green" if is_healthy else "conf-bar-red"
        badge_cls  = "badge-healthy" if is_healthy else "badge-unhealthy"
        keterangan = (
            "🟢 Buah cabai terdeteksi dalam kondisi <strong>sehat</strong>. "
            "Tidak ada indikasi kerusakan yang terdeteksi."
            if is_healthy else
            "🔴 Buah cabai terdeteksi dalam kondisi <strong>tidak sehat</strong>. "
            "Terdapat indikasi kerusakan atau penyakit pada buah cabai."
        )

        _, col_res, _ = st.columns([0.1, 0.8, 0.1])
        with col_res:

            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.6rem;
                        font-weight:700;font-size:1rem;color:#0d1f17;margin-bottom:1rem;">
              <div style="width:34px;height:34px;background:#00c896;border-radius:9px;
                          display:flex;align-items:center;justify-content:center;font-size:16px;">📊</div>
              Hasil Deteksi
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
              <div class="{badge_cls}">{emoji} {label_id}</div>
              <div style="font-size:1.35rem;font-weight:800;color:#0d1f17;margin-bottom:.25rem;">
                Hasil Klasifikasi
              </div>
              <div style="color:#5a7a6a;font-size:.85rem;">CNN EfficientNetB0 · Transfer Learning</div>
              <div style="margin-top:1rem;">
                <div style="display:flex;justify-content:space-between;
                            font-size:.82rem;color:#5a7a6a;margin-bottom:.25rem;">
                  <span>Tingkat Keyakinan Model</span>
                  <span style="font-weight:700;color:{color_hex}">{conf:.1f}%</span>
                </div>
                <div class="conf-bar-wrap">
                  <div class="{bar_class}" style="width:{conf:.1f}%"></div>
                </div>
              </div>
              <div class="metrics-row">
                <div class="metric-box">
                  <div class="metric-val">{conf:.1f}%</div>
                  <div class="metric-lbl">Keyakinan</div>
                </div>
                <div class="metric-box">
                  <div class="metric-val">{raw_score:.4f}</div>
                  <div class="metric-lbl">Raw Score</div>
                </div>
                <div class="metric-box">
                  <div class="metric-val" style="font-size:.88rem;">{"Healthy" if is_healthy else "Unhealthy"}</div>
                  <div class="metric-lbl">Label</div>
                </div>
              </div>
              <div style="margin-top:1.2rem;padding:0.9rem;
                          background:rgba(0,200,150,.07);border-radius:12px;
                          font-size:.83rem;color:#2d5a45;line-height:1.6;">
                {keterangan}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-weight:700;font-size:0.88rem;color:#0d1f17;margin-bottom:0.7rem;">
              📸 Perbandingan Gambar
            </div>
            """, unsafe_allow_html=True)

            col_orig, col_proc = st.columns(2)

            orig_b64 = img_to_b64(pil_img_final)
            proc_b64 = img_to_b64(img_processed)
            label_proses = "🪄 Setelah Remove BG + Resize 224×224" if remove_bg_on else "⚙️ Setelah Resize 224×224"

            with col_orig:
                st.markdown(f"""
                <div style="border:1.5px solid rgba(0,200,150,0.25);border-radius:14px;
                            background:rgba(0,200,150,0.04);padding:0.8rem;text-align:center;">
                  <div style="font-size:0.74rem;font-weight:700;color:#5a7a6a;
                              margin-bottom:0.6rem;text-align:left;">🖼️ Gambar Asli (Original)</div>
                  <img src="data:image/jpeg;base64,{orig_b64}"
                       style="max-width:100%;border-radius:10px;object-fit:contain;
                              max-height:220px;box-shadow:0 2px 10px rgba(0,40,20,0.1);">
                </div>
                """, unsafe_allow_html=True)

            with col_proc:
                st.markdown(f"""
                <div style="border:1.5px solid rgba(0,200,150,0.25);border-radius:14px;
                            background:rgba(0,200,150,0.04);padding:0.8rem;text-align:center;">
                  <div style="font-size:0.74rem;font-weight:700;color:#5a7a6a;
                              margin-bottom:0.6rem;text-align:left;">{label_proses}</div>
                  <img src="data:image/jpeg;base64,{proc_b64}"
                       style="max-width:100%;border-radius:10px;object-fit:contain;
                              max-height:220px;box-shadow:0 2px 10px rgba(0,40,20,0.1);">
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error saat analisis: {e}")

elif not has_image:
    _, col_info, _ = st.columns([0.1, 0.8, 0.1])
    with col_info:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;color:#7a9a8a;
                    font-size:.88rem;background:rgba(255,255,255,0.5);
                    border-radius:16px;border:1px dashed rgba(0,200,150,0.3);">
          📁 Belum ada gambar. Upload foto cabai di atas lalu klik
          <strong>Analisis Gambar</strong>.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="divider"></div>
<div class="footer">
  🌶️ <strong>CabaiDetect</strong> —
  Sistem Deteksi Buah Cabai Berbasis CNN EfficientNetB0<br>
  Skripsi 2026
</div>
""", unsafe_allow_html=True)
