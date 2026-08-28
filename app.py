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

# --- Theme System ---
from theme import inject_theme_css, render_theme_toggle, theme_color
inject_theme_css()

# --- Initialize Database & Auto-Sync Scheduler ---
from data.database import get_store
from data.scheduler import DataSyncScheduler

@st.cache_resource
def init_database():
    """Initialize SQLite database and migrate CSV data if needed."""
    store = get_store()
    csv_file = os.path.join(os.path.dirname(__file__), "food_prices_real.csv")
    if os.path.exists(csv_file):
        store.migrate_from_csv(csv_file)
    return store

@st.cache_resource
def init_scheduler(_store):
    """Start background scheduler for automatic daily PIHPS sync.
    
    Runs immediately on startup if data is stale (> 1 day behind),
    then repeats every 24 hours.
    """
    scheduler = DataSyncScheduler(_store)
    scheduler.start(interval_hours=24, run_immediately=True)
    return scheduler

store = init_database()
_scheduler = init_scheduler(store)

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
render_theme_toggle()
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
<div class="theme-accent-box">
    <h4>Selamat Datang di Agri-AI EWS v2.0</h4>
    <p>
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
    <div class="theme-card">
        <h4>🏠 Dashboard Utama</h4>
        <p>Forecast harga dengan confidence interval, EWS multi-faktor, dan analisis supply risk real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theme-card">
        <h4>🔬 Model Laboratory</h4>
        <p>Bandingkan performa Prophet, LSTM, dan TFT. Jalankan backtesting walk-forward.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="theme-card">
        <h4>🗺️ Regional Analysis</h4>
        <p>Peta choropleth interaktif Indonesia dengan drill-down per provinsi dan analisis disparitas.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="theme-card">
        <h4>📋 Reports</h4>
        <p>Generate laporan PDF dan Excel untuk download. Termasuk analisis EWS dan rekomendasi.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="theme-footer">
    <div>ENGINE: PROPHET v1.1 + PYTORCH BiLSTM + TFT</div>
    <div>DATA: PIHPS Bank Indonesia</div>
    <div>© 2026 Fahmi Prasanda</div>
</div>
""", unsafe_allow_html=True)
