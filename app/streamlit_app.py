"""Streamlit Dashboard Interaktif - Sistem Prediksi Harga Saham BBCA Berbasis Deep Learning & Sentimen Stockbit.

Fitur Utama:
- Dashboard Interaktif dengan Rich Aesthetics (Glassmorphic Cards, Glowing KPI Badges)
- Tab 1: Prediksi & Tren Harga Saham (Plotly Candlestick, Prediksi vs Aktual, Error Distribution)
- Tab 2: Analisis Sentimen Investor Stockbit (Rasio Sentimen, Korelasi Volume vs Return, Word Cloud)
- Tab 3: Perbandingan Model Deep Learning (LSTM, GRU, CNN 1D | Skenario S1 vs S2, Validasi Kappa)
- Tab 4: Fitur Teknikal & Data Explorer (RSI 14, Daily Return, Filter Data Table & Download CSV)
"""

import gzip
import html
import json
import sys
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
from wordcloud import WordCloud

# Pastikan path root terdaftar
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
from src.config import (
    CONFIG,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.features.assemble import prepare_scenario_data
from src.models.architectures import PyTorchSeqRegressor
from src.sentiment.preprocess_text import preprocess_text_full

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Prediksi Saham BBCA & Sentimen Stockbit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk Soft, Smooth, & Modern Fintech Aesthetics
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Premium Header Banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 30px 36px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(255, 255, 255, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.02rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }

    /* Header Info Chips */
    .header-chip {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 7px 16px;
        border-radius: 30px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.2s ease;
    }

    .header-chip:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-1px);
    }

    /* Metric Cards - Fixed Uniform Size & Alignment */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.01);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 160px;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 35px -10px rgba(99, 102, 241, 0.12);
        border-color: #c7d2fe;
    }

    .metric-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.6px;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0;
        letter-spacing: -0.5px;
    }

    /* Soft Status Badges */
    .metric-badge-positive {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 20px;
        background-color: #ecfdf5;
        color: #047857;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #a7f3d0;
    }

    .metric-badge-negative {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 20px;
        background-color: #fef2f2;
        color: #b91c1c;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #fecaca;
    }

    .metric-badge-neutral {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 20px;
        background-color: #f1f5f9;
        color: #475569;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #cbd5e1;
    }

    /* Horizontal Post Slider Container */
    .post-slider-container {
        display: flex;
        gap: 16px;
        overflow-x: auto;
        padding: 10px 4px 18px 4px;
        scroll-behavior: smooth;
        -webkit-overflow-scrolling: touch;
    }
    
    .post-slider-container::-webkit-scrollbar {
        height: 8px;
    }

    .post-slider-container::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }

    .post-slider-container::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }

    .post-slider-container::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    .post-slide-card {
        flex: 0 0 320px;
        min-width: 320px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        transition: all 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .post-slide-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(99, 102, 241, 0.1);
        border-color: #cbd5e1;
    }

    /* Academic Info Banner */
    .info-banner {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
        padding: 16px 20px;
        border-radius: 14px;
        margin-bottom: 24px;
        color: #0369a1;
        font-size: 0.92rem;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.05);
    }

    /* Smooth Custom Tab Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 22px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #64748b;
        border: none !important;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b;
        background-color: rgba(255, 255, 255, 0.5);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Custom Radio & Selectbox Styling */
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border-color: #cbd5e1 !important;
    }

    /* =========================================================
       RESPONSIVE MOBILE STYLES (Smartphones & Tablets < 768px)
       ========================================================= */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }

        /* Header Header Responsif */
        .main-header {
            padding: 20px 18px !important;
            border-radius: 18px !important;
            margin-bottom: 16px !important;
        }

        .main-header h1 {
            font-size: 1.4rem !important;
            line-height: 1.35 !important;
        }

        .main-header p {
            font-size: 0.88rem !important;
            line-height: 1.4 !important;
        }

        .header-chip {
            font-size: 0.78rem !important;
            padding: 5px 12px !important;
            width: 100% !important;
            box-sizing: border-border-box !important;
            display: block !important;
            text-align: center !important;
        }

        /* Status Banner Responsif Smartphone */
        .status-banner-container {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
            white-space: normal !important;
            padding: 12px 14px !important;
            border-radius: 12px !important;
        }

        .status-banner-container > div {
            width: 100% !important;
            justify-content: space-between !important;
        }

        /* Metric Cards Stack & Scaling */
        .metric-card {
            min-height: 135px !important;
            height: auto !important;
            padding: 15px 16px !important;
            margin-bottom: 10px !important;
            border-radius: 16px !important;
        }

        .metric-value {
            font-size: 1.55rem !important;
        }

        .metric-title {
            font-size: 0.76rem !important;
        }

        /* Tabs Scrollable Mobile */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            white-space: nowrap !important;
            flex-wrap: nowrap !important;
            padding: 4px !important;
            border-radius: 12px !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 14px !important;
            font-size: 0.82rem !important;
            flex-shrink: 0 !important;
        }

        /* Horizontal Slider Mobile Card Width */
        .post-slide-card {
            flex: 0 0 260px !important;
            min-width: 260px !important;
            padding: 14px !important;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Mengonversi DataFrame ke bytes CSV dengan caching memori agar hemat RAM."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def load_dataset(method="InSet Lexicon"):
    if "IndoBERT" in str(method):
        dataset_path = PROCESSED_DATA_DIR / "dataset_final_indobert.csv"
        if not dataset_path.exists():
            dataset_path = PROCESSED_DATA_DIR / "dataset_final.csv"
    else:
        dataset_path = PROCESSED_DATA_DIR / "dataset_final.csv"

    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path)
    df["Date"] = pd.to_datetime(df["Date"])

    if "discussion_volume" in df.columns:
        # Jika ada data ter-log, konversi balik ke angka postingan riil
        if df["discussion_volume"].dtype == float and df["discussion_volume"].max() <= 15:
            df["discussion_volume"] = np.expm1(df["discussion_volume"]).round().astype(int)
        else:
            df["discussion_volume"] = df["discussion_volume"].fillna(0).astype(int)

    return df


@st.cache_data
def load_stream_posts():
    gz_path = RAW_DATA_DIR / "stream_bbca.jsonl.gz"
    stream_path = RAW_DATA_DIR / "stream_bbca.jsonl"
    posts = []
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        posts.append(json.loads(line))
                    except Exception:
                        continue
    elif stream_path.exists():
        with open(stream_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        posts.append(json.loads(line))
                    except Exception:
                        continue
    return posts


@st.cache_data
def load_scored_stream_posts(method="InSet Lexicon"):
    if "IndoBERT" in str(method):
        scored_path = INTERIM_DATA_DIR / "scored_posts_indobert.csv.gz"
        if not scored_path.exists():
            scored_path = INTERIM_DATA_DIR / "scored_posts.csv.gz"
    else:
        scored_path = INTERIM_DATA_DIR / "scored_posts.csv.gz"

    if scored_path.exists():
        return pd.read_csv(scored_path)

    posts = load_stream_posts()
    if not posts:
        return pd.DataFrame()

    df_posts = pd.DataFrame(posts)

    # 1. Deduplikasi (Hapus ID & Teks Duplikat)
    if "id" in df_posts.columns:
        df_posts = df_posts.drop_duplicates(subset=["id"])
    df_posts = df_posts.drop_duplicates(subset=["content"]).copy()

    # 2. Preprocessing & Filter Spam/Noise
    from src.sentiment.preprocess_text import preprocess_text_full
    from src.sentiment.lexicon_scorer import InSetLexiconScorer

    cleaned = [preprocess_text_full(str(c)) for c in df_posts["content"]]
    df_posts["cleaned_content"] = cleaned

    # Filter postingan yang memiliki teks bersih minimal 3 karakter
    mask_clean = df_posts["cleaned_content"].astype(str).str.strip().str.len() >= 3
    df_posts = df_posts.loc[mask_clean].copy()

    # 3. Scoring InSet Lexicon
    scorer = InSetLexiconScorer()
    scores_res = [scorer.score_text(c) for c in df_posts["cleaned_content"]]
    df_posts["sentiment_score"] = [round(s["score"], 2) for s in scores_res]

    labels_map = {1: "Positif (+1)", -1: "Negatif (-1)", 0: "Netral (0)"}
    df_posts["sentiment_label"] = [labels_map.get(s["label"], "Netral (0)") for s in scores_res]

    cols_order = [
        "id",
        "created_at",
        "content",
        "cleaned_content",
        "sentiment_score",
        "sentiment_label",
        "like_count",
    ]
    cols_present = [c for c in cols_order if c in df_posts.columns]
    return df_posts[cols_present]


@st.cache_data
def load_raw_stream_posts_df():
    posts = load_stream_posts()
    if not posts:
        return pd.DataFrame()
    return pd.DataFrame(posts)


@st.cache_data
def load_metrics_tables():
    results_path = METRICS_DIR / "results.csv"
    validation_path = METRICS_DIR / "sentiment_validation.csv"

    df_results = pd.read_csv(results_path) if results_path.exists() else None
    df_validation = pd.read_csv(validation_path) if validation_path.exists() else None

    return df_results, df_validation


@st.cache_resource
def load_selected_model_and_scalers(
    model_choice: str = "Best Model (Terbaik)",
    scenario_choice: str = "Best Model (Otomatis)",
    sentiment_method: str = "InSet Lexicon (Rule-based)",
):
    # 1. Tentukan skenario aktif secara presisi
    if "S1" in scenario_choice:
        active_scenario = "S1"
    elif "S3" in scenario_choice:
        active_scenario = "S3"
    elif "S2" in scenario_choice:
        active_scenario = "S2"
    elif "Tanpa Sentimen" in sentiment_method:
        active_scenario = "S1"
    elif "IndoBERT" in sentiment_method:
        active_scenario = "S3"
    elif "InSet" in sentiment_method:
        active_scenario = "S2"
    else:
        best_meta_path = MODELS_DIR / "best_model_meta.json"
        if best_meta_path.exists():
            try:
                with open(best_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    active_scenario = meta.get("scenario", "S3").upper()
            except Exception:
                active_scenario = "S3"
        else:
            active_scenario = "S3"

    s_code = active_scenario.lower()

    # 2. Tentukan nama file model & scalers yang cocok dengan active_scenario
    m_code = "lstm" if "LSTM" in model_choice else ("gru" if "GRU" in model_choice else ("cnn" if "CNN" in model_choice else "best_model"))

    if m_code == "best_model":
        if active_scenario == "S3":
            model_filename = "best_model.pt"
            feature_scaler_filename = "best_scaler_features.pkl"
            target_scaler_filename = "best_scaler_target.pkl"
        else:
            model_filename = f"gru_{s_code}.pt"
            feature_scaler_filename = f"scaler_features_{s_code}.pkl"
            target_scaler_filename = f"scaler_target_{s_code}.pkl"
    else:
        model_filename = f"{m_code}_{s_code}.pt"
        feature_scaler_filename = f"scaler_features_{s_code}.pkl"
        target_scaler_filename = f"scaler_target_{s_code}.pkl"

    model_path = MODELS_DIR / model_filename
    feature_scaler_path = MODELS_DIR / feature_scaler_filename
    target_scaler_path = MODELS_DIR / target_scaler_filename

    if not model_path.exists():
        model_path = MODELS_DIR / "best_model.pt"
    if not feature_scaler_path.exists():
        feature_scaler_path = MODELS_DIR / f"scaler_features_{s_code}.pkl"
        if not feature_scaler_path.exists():
            feature_scaler_path = MODELS_DIR / "best_scaler_features.pkl"
    if not target_scaler_path.exists():
        target_scaler_path = MODELS_DIR / f"scaler_target_{s_code}.pkl"
        if not target_scaler_path.exists():
            target_scaler_path = MODELS_DIR / "best_scaler_target.pkl"

    if (
        not model_path.exists()
        or not feature_scaler_path.exists()
        or not target_scaler_path.exists()
    ):
        return None, None, None, active_scenario

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    input_dim = getattr(feature_scaler, "n_features_in_", 13)

    regressor = PyTorchSeqRegressor.from_file(
        filepath=model_path,
        input_dim=input_dim,
        seq_len=CONFIG["features"]["lookback"],
    )

    return regressor, feature_scaler, target_scaler, active_scenario


@st.cache_data
def load_all_word_frequencies():
    df_scored = load_scored_stream_posts()
    if df_scored.empty or "cleaned_content" not in df_scored.columns:
        return {}, {}, {}

    from collections import Counter
    from src.sentiment.preprocess_text import INDONESIAN_STOPWORDS

    custom_stopwords = set(INDONESIAN_STOPWORDS) | {
        "bbca", "saham", "ini", "yang", "akan", "ada", "dan", "di", "ke", "dari", "pada",
        "untuk", "bisa", "banyak", "lagi", "saya", "kamu", "anda", "dengan", "atau",
        "jadi", "sudah", "juga", "user", "stockbit", "hari", "bca", "nya", "aja", "biar",
        "dapat", "sama", "kalau", "kalo", "karna", "karena", "bisa", "terus", "ihsg", "tp"
    }

    counts_all = Counter()
    counts_pos = Counter()
    counts_neg = Counter()

    for cleaned, label_val in zip(df_scored["cleaned_content"], df_scored["sentiment_label"]):
        cleaned_str = str(cleaned)
        label_str = str(label_val)

        words = [w for w in cleaned_str.split() if w not in custom_stopwords and len(w) > 2]
        
        counts_all.update(words)
        if "Positif" in label_str or label_str == "1":
            counts_pos.update(words)
        elif "Negatif" in label_str or label_str == "-1":
            counts_neg.update(words)

    return dict(counts_all), dict(counts_pos), dict(counts_neg)


def generate_wordcloud(sentiment_filter="All"):
    counts_all, counts_pos, counts_neg = load_all_word_frequencies()

    if sentiment_filter == "Positif":
        freq_dict = dict(sorted(counts_pos.items(), key=lambda x: x[1], reverse=True)[:200])
    elif sentiment_filter == "Negatif":
        freq_dict = dict(sorted(counts_neg.items(), key=lambda x: x[1], reverse=True)[:200])
    else:
        freq_dict = dict(sorted(counts_all.items(), key=lambda x: x[1], reverse=True)[:200])

    if not freq_dict:
        freq_dict = {"cuan": 10, "untung": 8, "terbang": 6, "naik": 5}

    colormap_map = {
        "Positif": "YlGn",
        "Negatif": "OrRd",
        "All": "plasma"
    }

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="#0f172a",
        colormap=colormap_map.get(sentiment_filter, "plasma"),
        max_words=140,
        min_font_size=10,
        random_state=42,
        collocations=False,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f172a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig, sum(freq_dict.values())




def main():
    # Load Data Dataset Utama
    df = load_dataset()

    # Header Utama & Metadata Tanggal
    if df is not None and not df.empty:
        latest_date_str = pd.to_datetime(df["Date"].iloc[-1]).strftime("%d %B %Y")
        min_date_str = pd.to_datetime(df["Date"].iloc[0]).strftime("%d %B %Y")
        total_days = len(df)
    else:
        latest_date_str = "-"
        min_date_str = "-"
        total_days = 0

    st.markdown(
        f"""
        <div class="main-header">
            <h1>📈 Dashboard Prediksi Harga Saham BBCA & Analisis Sentimen Stockbit</h1>
            <p>Sistem Pemodelan Deep Learning Interaktif (StockLSTM, StockGRU, StockCNN 1D) Berbasis Fitur Teknikal & Sentimen InSet Lexicon</p>
            <div style="margin-top: 18px; display: flex; gap: 12px; flex-wrap: wrap;">
                <span class="header-chip">
                    📅 Data Terakhir: <b>{latest_date_str}</b>
                </span>
                <span class="header-chip">
                    📊 Rentang Periode Data: <b>{min_date_str}</b> s/d <b>{latest_date_str}</b> ({total_days} Hari Bursa)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Kontrol Sistem
    st.sidebar.title("⚙️ Kontrol Dashboard")
    st.sidebar.subheader("Pengaturan Model & Skenario")

    sentiment_method = "InSet Lexicon Based (Rule-based)"

    model_choice = st.sidebar.selectbox(
        "🤖 Pilih Arsitektur Model",
        options=["Best Model (Terbaik)", "StockLSTM", "StockGRU", "StockCNN 1D"],
        index=0,
        help="Pilih algoritma pemodelan Deep Learning yang ingin diuji."
    )

    scenario_choice = st.sidebar.selectbox(
        "📌 Pilih Skenario Fitur",
        options=[
            "Best Model (Otomatis)",
            "Skenario S1 (Teknikal Saja - 9 Fitur)",
            "Skenario S2 (Teknikal + Sentimen InSet Lexicon - 13 Fitur)"
        ],
        index=0,
        help="S1: Hanya Indikator Teknikal\nS2: Teknikal + Sentimen InSet Lexicon"
    )

    lookback = st.sidebar.slider("Window Timestep (Hari)", 10, 60, 30, step=5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Detail Skenario Eksperimen")
    st.sidebar.info(
        "**S1 (Baseline):** 9 Fitur Teknikal\n\n"
        "**S2 (+InSet Lexicon):** 9 Fitur Teknikal + 4 Fitur Sentimen InSet Lexicon (13 Fitur)"
    )

    # Load Data Dataset Utama Berdasarkan Metode Sentimen Terpilih
    df = load_dataset(method=sentiment_method)

    regressor, feature_scaler, target_scaler, active_scenario = load_selected_model_and_scalers(
        model_choice, scenario_choice, sentiment_method
    )

    # Memuat dataset yang 100% presisi sesuai skenario aktif
    if active_scenario == "S1":
        df = load_dataset(method="Tanpa Sentimen (S1 Baseline)")
    else:
        df = load_dataset(method="InSet Lexicon (Rule-based)")

    df_results, df_validation = load_metrics_tables()

    # Hitung Nilai KPI Utama
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    diff = latest_close - prev_close
    pct_change = (diff / prev_close) * 100

    data_scenario = prepare_scenario_data(scenario=active_scenario, lookback=lookback)
    X_test = data_scenario["X_test"]
    y_test = data_scenario["y_test"]
    test_dates = data_scenario["test_dates"]

    next_pred, pred_pct, mape, rmse, mae = 0.0, 0.0, 0.0, 0.0, 0.0
    y_test_actual, preds_actual = np.array([]), np.array([])

    if regressor is not None and len(X_test) > 0 and feature_scaler is not None:
        preds_scaled = regressor.predict(X_test)
        preds_actual = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Sinkronisasi Metrik Evaluasi Resmi dari Tabel Benchmark
        m_name = "LSTM" if "LSTM" in model_choice else ("CNN" if "CNN" in model_choice else "GRU")
        if df_results is not None:
            matched_row = df_results[
                (df_results["Model"].str.upper() == m_name) &
                (df_results["Scenario"] == active_scenario)
            ]
            if not matched_row.empty:
                mape = float(matched_row["MAPE (%)"].values[0])
                rmse = float(matched_row["RMSE (IDR)"].values[0])
                mae = float(matched_row["MAE (IDR)"].values[0])
            else:
                mape = np.mean(np.abs((y_test_actual - preds_actual) / y_test_actual)) * 100
                rmse = np.sqrt(np.mean((y_test_actual - preds_actual) ** 2))
                mae = np.mean(np.abs(y_test_actual - preds_actual))
        else:
            mape = np.mean(np.abs((y_test_actual - preds_actual) / y_test_actual)) * 100
            rmse = np.sqrt(np.mean((y_test_actual - preds_actual) ** 2))
            mae = np.mean(np.abs(y_test_actual - preds_actual))

        # Kalkulasi Prediksi H+1 Riil Esok Hari (Masa Depan setelah tanggal data terakhir)
        n_feats = getattr(feature_scaler, "n_features_in_", 13)
        from src.features.assemble import SENTIMENT_FEATURES, TECHNICAL_FEATURES
        feature_cols = TECHNICAL_FEATURES if n_feats == 9 else (TECHNICAL_FEATURES + SENTIMENT_FEATURES)

        if len(df) >= lookback and all(col in df.columns for col in feature_cols):
            df_prep = df.copy()
            if "discussion_volume" in df_prep.columns:
                df_prep["discussion_volume"] = np.log1p(df_prep["discussion_volume"].fillna(0))

            last_seq = df_prep[feature_cols].tail(lookback).values
            last_seq_scaled = feature_scaler.transform(last_seq).reshape(1, lookback, n_feats)
            future_pred_scaled = regressor.predict(last_seq_scaled)
            next_pred = float(target_scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()[0])
            pred_diff = next_pred - latest_close
            pred_pct = (pred_diff / latest_close) * 100
        else:
            next_pred = preds_actual[-1] if len(preds_actual) > 0 else latest_close
            pred_diff = next_pred - latest_close
            pred_pct = (pred_diff / latest_close) * 100

    # Format Nama Model & Skenario Ringkas 1 Baris
    if "Best Model" in model_choice:
        display_model_name = "StockGRU (Terbaik - MAPE 2,21%)"
    else:
        display_model_name = model_choice

    if active_scenario == "S1":
        display_scenario_name = "S1 (Teknikal Baseline - 9 Fitur)"
    else:
        display_scenario_name = "S2 (Teknikal + InSet Lexicon - 13 Fitur)"

    # Status Banner Compact 1 Baris Tunggal
    st.markdown(
        f"""
        <div class="status-banner-container" style="background: #eff6ff; border: 1px solid #bfdbfe; border-left: 5px solid #2563eb; padding: 10px 20px; border-radius: 14px; margin-bottom: 22px; display: flex; align-items: center; justify-content: space-between; gap: 16px; white-space: nowrap; overflow-x: auto; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.06);">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-weight: 700; color: #1e3a8a; font-size: 0.88rem;">🤖 Model Aktif:</span>
                <span style="background: #2563eb; color: #ffffff; padding: 3px 12px; border-radius: 16px; font-weight: 700; font-size: 0.83rem;">
                    {display_model_name}
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-weight: 700; color: #1e3a8a; font-size: 0.88rem;">📌 Skenario Fitur:</span>
                <span style="background: #0284c7; color: #ffffff; padding: 3px 12px; border-radius: 16px; font-weight: 700; font-size: 0.83rem;">
                    {display_scenario_name}
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-weight: 700; color: #1e3a8a; font-size: 0.88rem;">🧠 Sentimen:</span>
                <span style="background: #059669; color: #ffffff; padding: 3px 12px; border-radius: 16px; font-weight: 700; font-size: 0.83rem;">
                    InSet Lexicon
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Kartu KPI Utama Sesuai Ukuran Seragam
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Harga Penutupan Terakhir</div>
                <div class="metric-value">Rp {latest_close:,.0f}</div>
                <div>
                    <span class="{ 'metric-badge-positive' if diff >= 0 else 'metric-badge-negative' }">
                        { '+' if diff >= 0 else '' }{diff:,.0f} ({pct_change:+.2f}%)
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Prediksi Harga H+1</div>
                <div class="metric-value">Rp {next_pred:,.0f}</div>
                <div>
                    <span class="{ 'metric-badge-positive' if pred_pct >= 0 else 'metric-badge-negative' }">
                        Ekspektasi: {pred_pct:+.2f}%
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Akurasi Model (MAPE)</div>
                <div class="metric-value">{mape:.2f}%</div>
                <div>
                    <span style="color: #475569; font-size: 0.82rem; font-weight: 600;">Metrik Evaluasi Utama</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Eror Saham (RMSE & MAE)</div>
                <div class="metric-value">Rp {rmse:,.0f}</div>
                <div>
                    <span style="color: #475569; font-size: 0.82rem; font-weight: 600;">MAE: Rp {mae:,.0f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section Sampel Postingan Sentimen Stockbit di Dashboard Utama (Horizontal Slider)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💬 Sampel Live Postingan Sentimen Investor Stockbit ($BBCA)", expanded=True):
        df_scored_posts = load_scored_stream_posts()
        if not df_scored_posts.empty:
            df_scored_posts["dt"] = pd.to_datetime(df_scored_posts["created_at"], errors="coerce")
            max_d = df_scored_posts["dt"].max()

            # Filter postingan 5 hari terakhir dari tanggal terbaru data scraping
            min_5d = max_d - pd.Timedelta(days=5)
            df_recent = df_scored_posts[df_scored_posts["dt"] >= min_5d].copy()
            if df_recent.empty:
                df_recent = df_scored_posts.copy()

            # Filter postingan berkualitas tinggi (tanpa spam singkat)
            df_hq_posts = df_recent[
                (df_recent["content"].astype(str).str.strip().str.len() >= 25) &
                (~df_recent["content"].astype(str).str.match(r"^\$[A-Z\s\$]+$"))
            ].copy()

            if df_hq_posts.empty:
                df_hq_posts = df_recent

            samp_col1, samp_col2 = st.columns([1, 4])
            with samp_col1:
                sample_filter = st.radio(
                    "Filter Sentimen Sampel:",
                    ["Semua Label", "Positif (+1)", "Negatif (-1)", "Netral (0)"],
                    horizontal=False,
                    index=0,
                    key="main_sample_post_filter"
                )
                st.caption("👈 **Geser slider ke kanan/kiri** untuk melihat postingan terbaru (5 hari terakhir) yang akurat.")

            with samp_col2:
                # Sub-filter akurat berdasarkan label & ambang batas skor sentimen presisi
                if sample_filter == "Positif (+1)":
                    filtered_sample_df = df_hq_posts[
                        (df_hq_posts["sentiment_label"].str.contains("Positif")) &
                        (df_hq_posts["sentiment_score"] >= 1.0)
                    ].sort_values(by="created_at", ascending=False)
                elif sample_filter == "Negatif (-1)":
                    filtered_sample_df = df_hq_posts[
                        (df_hq_posts["sentiment_label"].str.contains("Negatif")) &
                        (df_hq_posts["sentiment_score"] <= -1.0)
                    ].sort_values(by="created_at", ascending=False)
                elif sample_filter == "Netral (0)":
                    filtered_sample_df = df_hq_posts[
                        (df_hq_posts["sentiment_label"].str.contains("Netral")) &
                        (df_hq_posts["sentiment_score"].abs() <= 0.5)
                    ].sort_values(by="created_at", ascending=False)
                else:
                    # Opsi "Semua Label": Ambil postingan terbaru paling akurat & selang-selingkan Positif, Negatif, Netral
                    pos_df = df_hq_posts[(df_hq_posts["sentiment_label"].str.contains("Positif")) & (df_hq_posts["sentiment_score"] >= 1.0)].sort_values(by="created_at", ascending=False).head(7)
                    neg_df = df_hq_posts[(df_hq_posts["sentiment_label"].str.contains("Negatif")) & (df_hq_posts["sentiment_score"] <= -1.0)].sort_values(by="created_at", ascending=False).head(7)
                    net_df = df_hq_posts[(df_hq_posts["sentiment_label"].str.contains("Netral")) & (df_hq_posts["sentiment_score"].abs() <= 0.5)].sort_values(by="created_at", ascending=False).head(6)

                    # Jika salah satu kategori sedikit pada 5 hari terakhir, fallback ke postingan terbaik terdekat
                    if pos_df.empty: pos_df = df_hq_posts[df_hq_posts["sentiment_label"].str.contains("Positif")].sort_values(by="created_at", ascending=False).head(7)
                    if neg_df.empty: neg_df = df_hq_posts[df_hq_posts["sentiment_label"].str.contains("Negatif")].sort_values(by="created_at", ascending=False).head(7)
                    if net_df.empty: net_df = df_hq_posts[df_hq_posts["sentiment_label"].str.contains("Netral")].sort_values(by="created_at", ascending=False).head(6)

                    mix_list = []
                    for i in range(max(len(pos_df), len(neg_df), len(net_df))):
                        if i < len(pos_df): mix_list.append(pos_df.iloc[i])
                        if i < len(neg_df): mix_list.append(neg_df.iloc[i])
                        if i < len(net_df): mix_list.append(net_df.iloc[i])
                    filtered_sample_df = pd.DataFrame(mix_list)

                if filtered_sample_df.empty:
                    filtered_sample_df = df_hq_posts.sort_values(by="created_at", ascending=False)

                top_sample_posts = filtered_sample_df.head(20).to_dict(orient="records")



                cards_html = []
                for post_item in top_sample_posts:
                    s_label = str(post_item.get("sentiment_label", "Netral (0)"))
                    if "Positif" in s_label:
                        badge_class = "metric-badge-positive"
                        badge_icon = "🟢"
                    elif "Negatif" in s_label:
                        badge_class = "metric-badge-negative"
                        badge_icon = "🔴"
                    else:
                        badge_class = "metric-badge-neutral"
                        badge_icon = "⚪"

                    post_time = str(post_item.get("created_at", ""))[:19].replace("T", " ")
                    raw_content = str(post_item.get("content", ""))
                    if len(raw_content) > 120:
                        raw_content = raw_content[:120] + "..."

                    clean_content = html.escape(raw_content.replace("\n", " ").replace("\r", " ").strip())
                    score_val = float(post_item.get("sentiment_score", 0.0))
                    likes_val = int(post_item.get("like_count", 0))

                    card_item_html = (
                        f'<div class="post-slide-card">'
                        f'<div>'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">'
                        f'<span class="{badge_class}">{badge_icon} {s_label}</span>'
                        f'<span style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">{post_time}</span>'
                        f'</div>'
                        f'<div style="font-size: 0.88rem; color: #1e293b; line-height: 1.55; margin-bottom: 12px; min-height: 65px; max-height: 75px; overflow: hidden;">'
                        f'"{clean_content}"'
                        f'</div>'
                        f'</div>'
                        f'<div style="font-size: 0.78rem; color: #64748b; font-weight: 600; display: flex; justify-content: space-between; border-top: 1px solid #f1f5f9; padding-top: 8px;">'
                        f'<span>❤️ {likes_val} Likes</span>'
                        f'<span>Skor: <b>{score_val:+.2f}</b></span>'
                        f'</div>'
                        f'</div>'
                    )
                    cards_html.append(card_item_html)

                slider_full_html = f'<div class="post-slider-container">{"".join(cards_html)}</div>'
                st.markdown(slider_full_html, unsafe_allow_html=True)



    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-Tab Dashboard Layout

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Prediksi & Tren Harga Saham",
            "💬 Analisis Sentimen Stockbit",
            "📊 Perbandingan Model & Metrik",
            "🛠️ Fitur Teknikal & Data Explorer",
        ]
    )

    # TAB 1: PREDIKSI & TREN HARGA SAHAM
    with tab1:
        st.subheader("📊 Visualisasi Tren Harga & Hasil Prediksi Model Deep Learning")

        # Interactive Zoom Controls
        zoom_col1, zoom_col2 = st.columns([2, 1])
        with zoom_col1:
            zoom_period = st.radio(
                "🔍 Zoom Filter Periode Grafik Test:",
                ["Semua Data Test (Full)", "6 Bulan Terakhir", "3 Bulan Terakhir"],
                horizontal=True,
                index=0,
                key="zoom_radio"
            )

        # Line Chart Harga Aktual vs Prediksi
        if len(y_test_actual) > 0:
            min_len = min(len(test_dates), len(y_test_actual), len(preds_actual))
            
            # Slice test dates & predictions according to zoom selection
            if zoom_period == "3 Bulan Terakhir":
                slice_n = min(60, min_len)
            elif zoom_period == "6 Bulan Terakhir":
                slice_n = min(120, min_len)
            else:
                slice_n = min_len

            plot_dates = test_dates[-slice_n:]
            plot_actual = y_test_actual[-slice_n:]
            plot_preds = preds_actual[-slice_n:]
            
            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=plot_actual,
                    mode="lines+markers",
                    name="Harga Penutupan Aktual (BBCA)",
                    line=dict(color="#1e40af", width=2.5),
                    marker=dict(size=4),
                )
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=plot_preds,
                    mode="lines+markers",
                    name=f"Prediksi {model_choice} ({active_scenario})",
                    line=dict(color="#f97316", width=2, dash="dash"),
                    marker=dict(size=4),
                )
            )

            fig_pred.update_layout(
                title=f"Perbandingan Harga Aktual vs Prediksi Model {model_choice} ({zoom_period})",
                xaxis_title="Tanggal",
                yaxis_title="Harga Penutupan (IDR)",
                hovermode="x unified",
                template="plotly_white",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Row 2: Error Residual Distribution & Error Scatter Plot
            col_err1, col_err2 = st.columns(2)
            residuals = plot_actual - plot_preds

            with col_err1:
                fig_hist = px.histogram(
                    x=residuals,
                    nbins=30,
                    title="Distribusi Residual Error Prediksi (Aktual - Prediksi)",
                    labels={"x": "Selisih Harga Error (IDR)", "y": "Frekuensi"},
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_hist.update_layout(template="plotly_white", height=380)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_err2:
                fig_scat = px.scatter(
                    x=plot_actual,
                    y=plot_preds,
                    title="Scatter Plot: Harga Aktual vs Harga Prediksi",
                    labels={"x": "Harga Aktual (IDR)", "y": "Harga Prediksi (IDR)"},
                    color_discrete_sequence=["#8b5cf6"],
                )
                fig_scat.add_shape(
                    type="line",
                    x0=min(plot_actual),
                    y0=min(plot_actual),
                    x1=max(plot_actual),
                    y1=max(plot_actual),
                    line=dict(color="gray", dash="dot"),
                )
                fig_scat.update_layout(template="plotly_white", height=380)
                st.plotly_chart(fig_scat, use_container_width=True)


        # Interactive Candlestick Chart
        st.markdown("---")
        st.markdown("### 🕯️ Grafik Candlestick & Overlays Indikator Teknikal Interaktif")
        
        candle_col1, candle_col2 = st.columns([2.5, 1])
        with candle_col1:
            selected_overlays = st.multiselect(
                "Pilih Indikator Overlay:",
                ["MA5 (Moving Average 5 Hari)", "MA20 (Moving Average 20 Hari)"],
                default=["MA5 (Moving Average 5 Hari)", "MA20 (Moving Average 20 Hari)"],
                key="candle_overlays"
            )
        with candle_col2:
            st.write("")
            st.write("")
            show_rangeslider = st.checkbox("Aktifkan Range Slider Zoom", value=True, key="range_slider_cb")

        # Set default visible range to last 6 months for clear spacing
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        six_m_ago = last_date - pd.DateOffset(months=6)

        fig_candle = go.Figure()
        fig_candle.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC BBCA",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
            )
        )
        if "MA5 (Moving Average 5 Hari)" in selected_overlays:
            fig_candle.add_trace(
                go.Scatter(x=df["Date"], y=df["MA5"], mode="lines", name="MA5", line=dict(color="#3b82f6", width=1.8))
            )
        if "MA20 (Moving Average 20 Hari)" in selected_overlays:
            fig_candle.add_trace(
                go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20", line=dict(color="#f97316", width=1.8))
            )

        fig_candle.update_layout(
            title="Pergerakan Harga Historis Saham BBCA (Interaktif OHLC & Moving Averages)",
            xaxis_title="Tanggal",
            yaxis_title="Harga Saham (IDR)",
            template="plotly_white",
            height=540,
            hovermode="x unified",
            xaxis=dict(
                range=[six_m_ago.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")],
                rangeslider=dict(visible=show_rangeslider, thickness=0.08),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    font=dict(size=12),
                    bgcolor="#f1f5f9",
                    activecolor="#cbd5e1"
                ),
                type="date"
            )
        )
        st.plotly_chart(fig_candle, use_container_width=True)



    # TAB 2: ANALISIS SENTIMEN STOCKBIT
    with tab2:
        st.subheader("💬 Analisis Sentimen Diskusi Investor Stockbit (InSet Lexicon)")

        # Section 1: Tren Harian & Korelasi Volume (Bagian Atas)
        col_sent1, col_sent2 = st.columns(2)

        with col_sent1:
            st.markdown("#### Tren Rasio Sentimen Positif vs Negatif Harian")
            fig_sent_line = go.Figure()
            fig_sent_line.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["positive_ratio"],
                    mode="lines",
                    name="Rasio Sentimen Positif",
                    line=dict(color="#10b981", width=2),
                )
            )
            fig_sent_line.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["negative_ratio"],
                    mode="lines",
                    name="Rasio Sentimen Negatif",
                    line=dict(color="#ef4444", width=2),
                )
            )
            fig_sent_line.update_layout(
                title="Proporsi Sentimen Investor Per Hari",
                xaxis_title="Tanggal",
                yaxis_title="Proporsi Rasio (0.0 - 1.0)",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_sent_line, use_container_width=True)

        with col_sent2:
            st.markdown("#### Korelasi Volume Diskusi dengan Daily Return Saham")
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(
                go.Bar(x=df["Date"], y=df["discussion_volume"], name="Volume Diskusi", marker_color="#94a3b8"),
                secondary_y=False,
            )
            fig_dual.add_trace(
                go.Scatter(x=df["Date"], y=df["Return"] * 100, name="Daily Return (%)", line=dict(color="#2563eb", width=1.5)),
                secondary_y=True,
            )
            fig_dual.update_layout(
                title="Volume Posting Diskusi vs Return Harian BBCA",
                template="plotly_white",
                height=380,
            )
            fig_dual.update_yaxes(title_text="Volume Diskusi (Post)", secondary_y=False)
            fig_dual.update_yaxes(title_text="Return Harian (%)", secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

        # Section 2: Donut Chart Distribusi & Sampel Postingan Terfilter Tanggal (Bagian Bawah)
        st.markdown("---")
        st.markdown("### 🍩 Analisis Distribusi Sentimen Interaktif & Sampel Postingan Per Tanggal")
        st.caption("Pilih rentang tanggal di bawah untuk melihat proporsi donut chart dan sampel postingan pada periode tertentu:")

        df_scored_posts = load_scored_stream_posts()
        if not df_scored_posts.empty:
            df_scored_posts["dt"] = pd.to_datetime(df_scored_posts["created_at"], errors="coerce")
            valid_dates = df_scored_posts["dt"].dropna().dt.date
            min_post_d = valid_dates.min() if not valid_dates.empty else df["Date"].iloc[0].date()
            max_post_d = valid_dates.max() if not valid_dates.empty else df["Date"].iloc[-1].date()

            range_col1, range_col2 = st.columns(2)
            with range_col1:
                sent_start_d = st.date_input("Tanggal Mulai Filter Sentimen", value=min_post_d, min_value=min_post_d, max_value=max_post_d, key="tab2_start_date")
            with range_col2:
                sent_end_d = st.date_input("Tanggal Selesai Filter Sentimen", value=max_post_d, min_value=min_post_d, max_value=max_post_d, key="tab2_end_date")

            # Filter data berdasarkan rentang tanggal
            mask_posts = (df_scored_posts["dt"].dt.date >= sent_start_d) & (df_scored_posts["dt"].dt.date <= sent_end_d)
            df_range_posts = df_scored_posts.loc[mask_posts]

            if df_range_posts.empty:
                df_range_posts = df_scored_posts

            dist_col1, dist_col2 = st.columns([1.3, 2.2])

            with dist_col1:
                st.markdown("#### 🍩 Proporsi Donut Chart Sentimen")
                cnt_pos = (df_range_posts["sentiment_label"].str.contains("Positif")).sum()
                cnt_neg = (df_range_posts["sentiment_label"].str.contains("Negatif")).sum()
                cnt_net = (df_range_posts["sentiment_label"].str.contains("Netral")).sum()
                total_range_posts = len(df_range_posts)

                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Positif (+1)", "Negatif (-1)", "Netral (0)"],
                            values=[cnt_pos, cnt_neg, cnt_net],
                            hole=0.45,
                            marker_colors=["#10b981", "#ef4444", "#94a3b8"],
                            textinfo="label+percent",
                            hoverinfo="label+value+percent",
                        )
                    ]
                )
                fig_pie.update_layout(
                    title=f"Distribusi Sentimen ({sent_start_d} s/d {sent_end_d})",
                    template="plotly_white",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=True,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption(f"📊 **Total Post Terfilter**: **{total_range_posts:,}** | 🟢 Positif: **{cnt_pos:,}** | 🔴 Negatif: **{cnt_neg:,}** | ⚪ Netral: **{cnt_net:,}**")

            with dist_col2:
                st.markdown("#### 💬 Sampel Postingan Stockbit Terfilter Tanggal")
                sample_range_filter = st.radio(
                    "Filter Label Sampel:",
                    ["Semua Label", "Positif (+1)", "Negatif (-1)", "Netral (0)"],
                    horizontal=True,
                    key="tab2_sample_radio"
                )
                if sample_range_filter != "Semua Label":
                    df_sample_sub = df_range_posts[df_range_posts["sentiment_label"] == sample_range_filter]
                else:
                    df_sample_sub = df_range_posts

                if df_sample_sub.empty:
                    df_sample_sub = df_range_posts

                top_sample_range = df_sample_sub.head(10).to_dict(orient="records")

                sample_cards_html = []
                for p_item in top_sample_range:
                    lbl = str(p_item.get("sentiment_label", "Netral (0)"))
                    b_cls = "metric-badge-positive" if "Positif" in lbl else ("metric-badge-negative" if "Negatif" in lbl else "metric-badge-neutral")
                    b_ico = "🟢" if "Positif" in lbl else ("🔴" if "Negatif" in lbl else "⚪")
                    p_t = str(p_item.get("created_at", ""))[:19].replace("T", " ")
                    r_c = str(p_item.get("content", ""))
                    if len(r_c) > 110:
                        r_c = r_c[:110] + "..."
                    c_c = html.escape(r_c.replace("\n", " ").replace("\r", " ").strip())
                    sc = float(p_item.get("sentiment_score", 0.0))
                    lk = int(p_item.get("like_count", 0))

                    card_h = (
                        f'<div class="post-slide-card">'
                        f'<div>'
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
                        f'<span class="{b_cls}">{b_ico} {lbl}</span>'
                        f'<span style="font-size: 0.72rem; color: #94a3b8;">{p_t}</span>'
                        f'</div>'
                        f'<div style="font-size: 0.85rem; color: #1e293b; line-height: 1.45; min-height: 55px; max-height: 65px; overflow: hidden;">'
                        f'"{c_c}"'
                        f'</div>'
                        f'</div>'
                        f'<div style="font-size: 0.75rem; color: #64748b; font-weight: 600; display: flex; justify-content: space-between; border-top: 1px solid #f1f5f9; padding-top: 6px;">'
                        f'<span>❤️ {lk} Likes</span><span>Skor: <b>{sc:+.2f}</b></span>'
                        f'</div>'
                        f'</div>'
                    )
                    sample_cards_html.append(card_h)

                st.markdown(f'<div class="post-slider-container">{"".join(sample_cards_html)}</div>', unsafe_allow_html=True)

        # Word Cloud Section
        st.markdown("---")
        st.markdown("### ☁️ Word Cloud Visualisasi Sentimen Stockbit")
        
        wc_col1, wc_col2 = st.columns([1, 2.8])
        with wc_col1:
            wc_filter = st.radio(
                "Filter Kategori Sentimen:",
                ["All", "Positif", "Negatif"],
                horizontal=False,
                index=0,
                key="wc_sentiment_radio"
            )
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; padding: 15px; border-radius: 10px; border-left: 4px solid #2563eb; margin-top: 10px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b;">📌 Informasi Word Cloud</div>
                    <div style="font-size: 0.8rem; color: #475569; margin-top: 5px;">
                        Visualisasi diolah secara terstruktur dari <b>keseluruhan 255.000+ postingan Stockbit (2021-2026)</b> yang diklasifikasikan dengan InSet Lexicon & dibersihkan dari kata bising.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with wc_col2:
            with st.spinner("Menggenerasi Word Cloud dari 255.000+ data Stockbit..."):
                fig_wc, total_w = generate_wordcloud(sentiment_filter=wc_filter)
                st.pyplot(fig_wc)



    # TAB 3: PERBANDINGAN MODEL & METRIK
    with tab3:
        st.subheader("📊 Evaluasi Perbandingan Arsitektur Deep Learning & Parameter Tuning (K1 - K5)")

        # 1. Tabel Referensi Hyperparameter Tuning (K1 s/d K5)
        st.markdown("### ⚙️ Spesifikasi Parameter Tuning Hyperparameter (K1 s/d K5)")
        st.caption("Berikut adalah konfigurasi kombinasi hyperparameter yang diuji untuk menemukan arsitektur terbaik:")
        
        tuning_params_data = [
            {"Config": "K1", "Hidden Units / Filters": 32, "Dropout": 0.10, "Learning Rate": 0.0050, "Batch Size": 16, "Epochs": 50},
            {"Config": "K2", "Hidden Units / Filters": 64, "Dropout": 0.20, "Learning Rate": 0.0010, "Batch Size": 32, "Epochs": 100},
            {"Config": "K3", "Hidden Units / Filters": 64, "Dropout": 0.20, "Learning Rate": 0.0010, "Batch Size": 16, "Epochs": 150},
            {"Config": "K4", "Hidden Units / Filters": 128, "Dropout": 0.30, "Learning Rate": 0.0005, "Batch Size": 16, "Epochs": 100},
            {"Config": "K5", "Hidden Units / Filters": 64, "Dropout": 0.50, "Learning Rate": 0.0050, "Batch Size": 32, "Epochs": 200},
        ]
        st.dataframe(pd.DataFrame(tuning_params_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 2. Perbandingan Bar Chart & Ringkasan Metrik
        if df_results is not None:
            df_full_metrics = df_results[df_results["Scenario"].isin(["S1", "S2"])].copy()
            if "RMSE (IDR)" in df_full_metrics.columns and "MSE (IDR²)" not in df_full_metrics.columns:
                df_full_metrics["MSE (IDR²)"] = (df_full_metrics["RMSE (IDR)"] ** 2).apply(lambda x: round(x, 2))

            col_m1, col_m2 = st.columns([1.2, 1])

            with col_m1:
                st.markdown("#### Bar Chart Perbandingan MAPE (%) per Skenario (Best Config K3)")
                fig_comp = px.bar(
                    df_full_metrics,
                    x="Model",
                    y="MAPE (%)",
                    color="Scenario",
                    barmode="group",
                    text="MAPE (%)",
                    title="Perbandingan MAPE: S1 (Teknikal Baseline) vs S2 (InSet Lexicon)",
                    color_discrete_map={"S1": "#64748b", "S2": "#2563eb"},
                )
                fig_comp.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_comp, use_container_width=True)

            with col_m2:
                st.markdown("#### Ringkasan Evaluasi Metrik Utama (K3)")
                st.dataframe(df_full_metrics, use_container_width=True, hide_index=True)

            # 3. Tabel Komprehensif Seluruh Run (K1 s/d K5 untuk S1 dan S2)
            st.markdown("---")
            st.markdown("### 📋 Tabel Komprehensif Hasil Evaluation Run Seluruh Config (K1 - K5)")
            st.caption("Menampilkan perbandingan komprehensif metrik evaluasi (MAPE, RMSE, MAE, dan Stabilitas StdDev) untuk skenario S1 (Teknikal) vs S2 (InSet Lexicon):")

            # Memuat Data Hasil Run Nyata (Dynamic Loading)
            all_configs_csv = BASE_DIR / "reports" / "metrics" / "results_all_configs.csv"
            
            if all_configs_csv.exists():
                df_all_runs = pd.read_csv(all_configs_csv)
                if "RMSE (IDR)" in df_all_runs.columns and "MSE (IDR²)" not in df_all_runs.columns:
                    df_all_runs["MSE (IDR²)"] = (df_all_runs["RMSE (IDR)"] ** 2).apply(lambda x: round(x, 2))
                
                # Filter interactive untuk tabel komprehensif
                filter_c1, filter_c2 = st.columns(2)
                with filter_c1:
                    selected_model_tbl = st.multiselect("Filter Model Tabel:", ["LSTM", "GRU", "CNN"], default=["LSTM", "GRU", "CNN"])
                with filter_c2:
                    selected_scenario_tbl = st.multiselect("Filter Skenario Tabel:", ["S1", "S2"], default=["S1", "S2"])

                df_filtered_runs = df_all_runs[
                    (df_all_runs["Model"].isin(selected_model_tbl)) &
                    (df_all_runs["Scenario"].isin(selected_scenario_tbl))
                ]

                st.dataframe(df_filtered_runs, use_container_width=True, hide_index=True)

                # Tombol Download Data Hasil Benchmark
                csv_bytes = convert_df_to_csv(df_all_runs)
                st.download_button(
                    label="📥 Download Data Lengkap Hasil Benchmark (CSV)",
                    data=csv_bytes,
                    file_name="hasil_benchmark_skripsi_45_runs.csv",
                    mime="text/csv",
                    help="Klik untuk mengunduh seluruh data metrik evaluasi 45 kombinasi eksperimen untuk lampiran Bab 4."
                )
            else:
                st.info("ℹ️ File `results_all_configs.csv` belum ditemukan. Menampilkan hasil run optimal saat ini (K3):")
                st.dataframe(df_full_metrics, use_container_width=True, hide_index=True)

                csv_k3 = convert_df_to_csv(df_full_metrics)
                st.download_button(
                    label="📥 Download Ringkasan Evaluasi K3 (CSV)",
                    data=csv_k3,
                    file_name="hasil_evaluasi_k3.csv",
                    mime="text/csv"
                )


    # TAB 4: FITUR TEKNIKAL & DATA EXPLORER
    with tab4:
        st.subheader("🛠️ Indikator Teknikal & Interactive Data Explorer")

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(
            [
                "📋 Dataset Final (Harga + Sentimen)",
                "✨ Data Final Cleaning Stockbit (Cleaned Data)",
                "💬 Data Scraping Stockbit Asli (Raw Stream)",
            ]
        )

        with sub_tab1:
            col_tech1, col_tech2 = st.columns(2)

            with col_tech1:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI 14", line=dict(color="#8b5cf6", width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
                fig_rsi.update_layout(
                    title="Relative Strength Index (RSI 14)",
                    xaxis_title="Tanggal",
                    yaxis_title="Nilai RSI",
                    template="plotly_white",
                    height=380,
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

            with col_tech2:
                # High Contrast Dual-Color Volume Bar Chart (Green = Bullish, Red = Bearish)
                df_vol = df.copy()
                vol_colors = np.where(df_vol["Close"] >= df_vol["Open"], "#10b981", "#ef4444")

                fig_vol = go.Figure()
                fig_vol.add_trace(
                    go.Bar(
                        x=df_vol["Date"],
                        y=df_vol["Volume"],
                        name="Volume Transaksi",
                        marker_color=vol_colors,
                        marker_line_width=0,
                    )
                )
                fig_vol.update_layout(
                    title="Volume Perdagangan Saham BBCA (Kontras Tinggi: Green = Bull, Red = Bear)",
                    xaxis_title="Tanggal",
                    yaxis_title="Volume Transaksi (Lembar Saham)",
                    template="plotly_white",
                    height=380,
                )
                st.plotly_chart(fig_vol, use_container_width=True)


            # Interactive Data Table & Download CSV Button
            st.markdown("---")
            st.markdown("### 📋 Filter Data Table & Download Dataset Final")

            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                start_date = st.date_input("Tanggal Mulai Filter", value=df["Date"].min())
            with col_filter2:
                end_date = st.date_input("Tanggal Selesai Filter", value=df["Date"].max())

            mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask]

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            csv_data = convert_df_to_csv(filtered_df)
            st.download_button(
                label="📥 Download Data CSV Terfilter",
                data=csv_data,
                file_name="dataset_bbca_filtered.csv",
                mime="text/csv",
            )

        with sub_tab2:
            st.markdown("### ✨ Data Final Cleaning Stockbit ($BBCA 2021-2026)")
            st.caption("Data postingan bersih setelah melalui tahap deduplikasi, filter spam multi-ticker, preprocessing teks lengkap, dan pelabelan InSet Lexicon:")

            df_scored_posts = load_scored_stream_posts()

            if not df_scored_posts.empty:
                # Filter data bersih (tanpa spam & noise singkat)
                df_clean_all = df_scored_posts[
                    (df_scored_posts["content"].astype(str).str.strip().str.len() >= 3)
                ].copy()

                min_clean_d = str(df_clean_all["created_at"].min())[:10]
                max_clean_d = str(df_clean_all["created_at"].max())[:10]

                col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                with col_c1:
                    st.metric("Total Postingan Bersih", f"{len(df_clean_all):,} Postingan")
                with col_c2:
                    st.metric("Rentang Tanggal Bersih", f"{min_clean_d} s/d {max_clean_d}")
                with col_c3:
                    c_pos = (df_clean_all["sentiment_label"] == "Positif (+1)").sum()
                    st.metric("Sentimen Positif (+1)", f"{c_pos:,}", f"{(c_pos/len(df_clean_all))*100:.1f}%")
                with col_c4:
                    c_neu = (df_clean_all["sentiment_label"] == "Netral (0)").sum()
                    st.metric("Sentimen Netral (0)", f"{c_neu:,}", f"{(c_neu/len(df_clean_all))*100:.1f}%")
                with col_c5:
                    c_neg = (df_clean_all["sentiment_label"] == "Negatif (-1)").sum()
                    st.metric("Sentimen Negatif (-1)", f"{c_neg:,}", f"{(c_neg/len(df_clean_all))*100:.1f}%")

                col_cf1, col_cf2, col_cf3 = st.columns(3)
                with col_cf1:
                    kw_clean = st.text_input("🔍 Cari Kata Kunci pada Postingan Bersih", "", key="kw_clean_input")
                with col_cf2:
                    lbl_clean = st.selectbox(
                        "🏷️ Filter Label Sentimen Bersih",
                        ["Semua Label", "Positif (+1)", "Negatif (-1)", "Netral (0)"],
                        index=0,
                        key="lbl_clean_select"
                    )
                with col_cf3:
                    sort_clean = st.selectbox(
                        "📅 Urutan Tanggal Tampilan",
                        ["Terbaru -> Terlama", "Terlama -> Terbaru"],
                        index=0,
                        key="sort_clean_select"
                    )

                df_view_clean = df_clean_all.copy()
                if kw_clean:
                    df_view_clean = df_view_clean[
                        df_view_clean["content"].astype(str).str.contains(kw_clean, case=False, na=False) |
                        df_view_clean["cleaned_content"].astype(str).str.contains(kw_clean, case=False, na=False)
                    ]
                if lbl_clean != "Semua Label":
                    df_view_clean = df_view_clean[df_view_clean["sentiment_label"] == lbl_clean]

                if sort_clean == "Terlama -> Terbaru":
                    df_view_clean = df_view_clean.sort_values(by="created_at", ascending=True)
                else:
                    df_view_clean = df_view_clean.sort_values(by="created_at", ascending=False)

                st.dataframe(df_view_clean, use_container_width=True, hide_index=True)

                csv_clean_data = convert_df_to_csv(df_view_clean)
                st.download_button(
                    label="📥 Download Data Final Cleaning Stockbit (CSV Terlabel & Terpreproses)",
                    data=csv_clean_data,
                    file_name="stockbit_final_cleaned_119k.csv",
                    mime="text/csv",
                )

        with sub_tab3:
            st.markdown("### 💬 Data Scraping Hasil Stream Diskusi Stockbit Mentah ($BBCA 2021-2026)")
            df_raw_posts = load_raw_stream_posts_df()

            if not df_raw_posts.empty:
                min_d = str(df_raw_posts["created_at"].min())[:10]
                max_d = str(df_raw_posts["created_at"].max())[:10]

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Total Postingan Raw Scraped (Mentah)", f"{len(df_raw_posts):,} Postingan")
                with col_p2:
                    st.metric("Jangkauan Tanggal Terkumpul", f"{min_d} s/d {max_d}")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    search_kw = st.text_input("🔍 Cari Kata Kunci pada Postingan Mentah", "", key="kw_raw_input")
                with col_f2:
                    sort_order = st.selectbox(
                        "📅 Urutan Tanggal Tampilan",
                        ["Terbaru -> Terlama", "Terlama -> Terbaru"],
                        index=0,
                        key="raw_sort_select"
                    )

                df_filtered = df_raw_posts.copy()
                if search_kw:
                    df_filtered = df_filtered[
                        df_filtered["content"].astype(str).str.contains(search_kw, case=False, na=False)
                    ]

                if sort_order == "Terlama -> Terbaru":
                    df_filtered = df_filtered.sort_values(by="created_at", ascending=True)
                else:
                    df_filtered = df_filtered.sort_values(by="created_at", ascending=False)


                st.dataframe(df_filtered, use_container_width=True, hide_index=True)

                raw_csv = convert_df_to_csv(df_filtered)
                st.download_button(
                    label="📥 Download Data Stream Stockbit CSV (Termasuk Skor & Label)",
                    data=raw_csv,
                    file_name="stream_stockbit_bbca_scored.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "Data scraping Stockbit mentah (`data/raw/stream_bbca.jsonl`) belum ditemukan. "
                    "Pastikan `STOCKBIT_BEARER_TOKEN` dikonfigurasi di file `.env` untuk menjalankan crawler."
                )



if __name__ == "__main__":
    main()
