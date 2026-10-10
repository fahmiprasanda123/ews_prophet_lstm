"""
Beranda: ringkasan data dan pintu masuk ke setiap halaman.
"""
import streamlit as st

from data.database import get_store
from theme import inject_theme_css, render_theme_toggle, render_sidebar_brand, render_footer

st.set_page_config(page_title="Agri-AI EWS | Indonesia Food Security", page_icon="🌾", layout="wide")
inject_theme_css()

store = get_store()

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

# --- Sidebar ---
render_sidebar_brand()
render_theme_toggle()
st.sidebar.divider()

# Show DB stats
stats = store.get_stats()
st.sidebar.markdown("**Status database**")
if stats['total_records'] > 0:
    st.sidebar.success(f"{stats['total_records']:,} baris data harga")
    st.sidebar.caption(f"Periode {stats['date_from']} s.d. {stats['date_to']}")
else:
    st.sidebar.warning("Belum ada data. Jalankan sinkronisasi PIHPS atau sediakan food_prices_real.csv.")

# --- Main content ---
st.title("Agri-AI Early Warning System")
st.markdown("Peringatan dini harga pangan Indonesia")

st.markdown(f"""
<div class="panel">
    <p>
        Memantau dan memprakirakan harga <strong>{stats['commodities']} komoditas pangan</strong>
        di <strong>{stats['provinces']} provinsi</strong> dari data PIHPS Bank Indonesia,
        dengan model Prophet, Bidirectional LSTM, dan Temporal Fusion Transformer
        yang digabung dalam satu ensemble.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick stats (tanpa delta: angka ini bukan tren)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baris data", f"{stats['total_records']:,}")
c2.metric("Provinsi", f"{stats['provinces']}")
c3.metric("Komoditas", f"{stats['commodities']}")
c4.metric("Data terakhir", stats.get('date_to') or "Belum ada")

st.divider()

st.subheader("Mulai dari sini")
st.page_link("pages/1_Dashboard.py", label="Dashboard: prakiraan harga dan status peringatan per provinsi")
st.page_link("pages/2_Analisis_Regional.py", label="Analisis Regional: peta harga dan disparitas antarprovinsi")
st.page_link("pages/3_Laboratorium_Model.py", label="Laboratorium Model: bandingkan Prophet, BiLSTM, TFT, dan backtesting")
st.page_link("pages/4_Laporan.py", label="Laporan: unduh PDF dan Excel")
st.page_link("pages/6_Chatbot.py", label="Chatbot: tanya harga dalam bahasa sehari-hari")

st.divider()
render_footer("Model: Prophet, BiLSTM, TFT", "Data: PIHPS Bank Indonesia", "© 2026 Fahmi Prasanda")
