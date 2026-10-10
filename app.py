"""
Agri-AI EWS v2.0: Multi-Page Streamlit Application Entry Point.

Router st.navigation: menjalankan setup bersama (tema, database, scheduler) lalu halaman terpilih.
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

# --- Navigation ---
# Label navigasi ditetapkan di sini; tanpa ini Streamlit memakai nama file "app".
pages = [
    st.Page("pages/0_Beranda.py", title="Beranda", default=True),
    st.Page("pages/1_Dashboard.py", title="Dashboard", url_path="Dashboard"),
    st.Page("pages/2_Analisis_Regional.py", title="Analisis Regional", url_path="Analisis_Regional"),
    st.Page("pages/3_Laboratorium_Model.py", title="Laboratorium Model", url_path="Laboratorium_Model"),
    st.Page("pages/4_Laporan.py", title="Laporan", url_path="Laporan"),
    st.Page("pages/5_Tentang.py", title="Tentang", url_path="Tentang"),
    st.Page("pages/6_Chatbot.py", title="Chatbot", url_path="Chatbot"),
]
st.navigation(pages).run()
