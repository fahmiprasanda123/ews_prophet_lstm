"""
Agri-AI EWS v2.0 — Multi-Page Streamlit Application Entry Point.
"""
import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Page Configuration ---
st.set_page_config(
    page_title="Agri-AI EWS | Indonesia Food Security",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global CSS ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1a1c24 100%);
        color: #E0E0E0;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        overflow-wrap: break-word;
        white-space: normal !important;
    }
    .warning-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
        font-size: 1.1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    header {visibility: hidden;}
    .factor-bar {
        height: 8px;
        border-radius: 4px;
        margin: 2px 0;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Database ---
from data.database import get_store

@st.cache_resource
def init_database():
    """Initialize SQLite database and migrate CSV data if needed."""
    store = get_store()
    csv_file = os.path.join(os.path.dirname(__file__), "food_prices_real.csv")
    if os.path.exists(csv_file):
        store.migrate_from_csv(csv_file)
    return store

store = init_database()

# --- Initialize Global Session State for Model Parameters ---
if 'model_params' not in st.session_state:
    st.session_state.model_params = {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'epochs': 10,
        'hidden_size': 128,
        'seq_length': 30,
        'tft_max_epochs': 2,
        'tft_batch_size': 32
    }

# --- Landing Page ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2534/2534044.png", width=60)
st.sidebar.title("🌾 Agri-AI EWS v2.0")
st.sidebar.caption("Early Warning System for Food Security")
st.sidebar.markdown("---")

# Show DB stats
stats = store.get_stats()
st.sidebar.write("📊 **Database Status**")
if stats['total_records'] > 0:
    st.sidebar.success(f"✅ {stats['total_records']:,} records loaded")
    st.sidebar.caption(f"📅 {stats['date_from']} → {stats['date_to']}")
else:
    st.sidebar.warning("⚠️ No data loaded yet")

st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the menu above ↑")

# --- Main content ---
st.title("🌾 Agri-AI Early Warning System v2.0")
st.markdown("### Platform Peringatan Dini Harga Pangan Indonesia")

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79,172,254,0.15) 0%, rgba(0,242,254,0.08) 100%);
    border: 1px solid rgba(79,172,254,0.3);
    border-radius: 16px;
    padding: 30px;
    margin: 20px 0;
">
    <h4 style="color: #4facfe; margin-top: 0;">Selamat Datang di Agri-AI EWS v2.0</h4>
    <p style="color: #ccc; font-size: 1.05rem;">
        Sistem peringatan dini berbasis AI untuk memantau dan memprediksi harga 
        <strong>10 komoditas pangan strategis</strong> di <strong>34 provinsi Indonesia</strong>, 
        menggunakan hybrid model <strong>Prophet + Bidirectional LSTM + Temporal Fusion Transformer</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Data", f"{stats['total_records']:,}", "records")
c2.metric("🗺️ Provinsi", f"{stats['provinces']}", "provinsi")
c3.metric("🌽 Komoditas", f"{stats['commodities']}", "komoditas strategis")
c4.metric("📅 Rentang Data", stats.get('date_to', 'N/A'), stats.get('date_from', ''))

st.markdown("---")

# Navigation cards
st.markdown("### 📌 Fitur Utama")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid rgba(255,255,255,0.1);">
        <h4>🏠 Dashboard Utama</h4>
        <p style="color: #aaa;">Forecast harga dengan confidence interval, EWS multi-faktor, dan analisis supply risk real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid rgba(255,255,255,0.1);">
        <h4>🔬 Model Laboratory</h4>
        <p style="color: #aaa;">Bandingkan performa Prophet, LSTM, dan TFT. Jalankan backtesting walk-forward.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid rgba(255,255,255,0.1);">
        <h4>🗺️ Regional Analysis</h4>
        <p style="color: #aaa;">Peta choropleth interaktif Indonesia dengan drill-down per provinsi dan analisis disparitas.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 8px 0; border: 1px solid rgba(255,255,255,0.1);">
        <h4>📋 Reports</h4>
        <p style="color: #aaa;">Generate laporan PDF dan Excel untuk download. Termasuk analisis EWS dan rekomendasi.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; opacity: 0.5; font-size: 0.8rem;">
    <div>ENGINE: PROPHET v1.1 + PYTORCH BiLSTM + TFT</div>
    <div>DATA: PIHPS Bank Indonesia</div>
    <div>© 2026 Fahmi Prasanda</div>
</div>
""", unsafe_allow_html=True)
