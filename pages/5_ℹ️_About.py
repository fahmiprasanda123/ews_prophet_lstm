"""
Page 5: About — Detailed explanation of the system, data sources, and methodology.
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="About | Agri-AI EWS", page_icon="ℹ️", layout="wide")

# --- Header ---
st.title("ℹ️ Tentang Agri-AI EWS v2.0")
st.markdown("### Dokumentasi Lengkap Sistem Peringatan Dini Harga Pangan Indonesia")

st.markdown("---")

# =====================================================================
# 1. OVERVIEW
# =====================================================================
st.markdown("## 🌾 Apa itu Agri-AI EWS?")
st.markdown("""
**Agri-AI Early Warning System (EWS)** adalah platform berbasis kecerdasan buatan 
untuk memantau, memprediksi, dan memberikan peringatan dini terhadap perubahan harga 
**10 komoditas pangan strategis** di **34 provinsi Indonesia**.

Sistem ini menggabungkan:
- 🧠 **3 model AI** (Prophet, Bidirectional LSTM, Temporal Fusion Transformer)
- 📊 **Data harga real-time** dari PIHPS Bank Indonesia
- 🌦️ **Data cuaca real** dari Open-Meteo + NOAA
- ⚠️ **Engine peringatan dini multi-faktor** untuk deteksi risiko
""")

# =====================================================================
# 2. DATA SOURCES
# =====================================================================
st.markdown("---")
st.markdown("## 📦 Sumber Data")

# Tabs for each data source
ds_tab1, ds_tab2, ds_tab3 = st.tabs(["💰 Data Harga", "🌦️ Data Cuaca", "🌊 Data ENSO"])

with ds_tab1:
    st.markdown("### 💰 Data Harga Pangan — PIHPS Bank Indonesia")

    st.markdown("""
    <div style="
        background: rgba(79,172,254,0.08);
        border: 1px solid rgba(79,172,254,0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    ">
        <table style="width: 100%; color: #ddd;">
            <tr><td style="width: 160px;"><strong>📡 Sumber</strong></td><td>Pusat Informasi Harga Pangan Strategis Nasional (PIHPS), Bank Indonesia</td></tr>
            <tr><td><strong>🌐 Website</strong></td><td><a href="https://www.bi.go.id/hargapangan" target="_blank">https://www.bi.go.id/hargapangan</a></td></tr>
            <tr><td><strong>📅 Rentang</strong></td><td>Januari 2021 — Sekarang (update harian pada hari kerja)</td></tr>
            <tr><td><strong>🗺️ Cakupan</strong></td><td>34 Provinsi seluruh Indonesia</td></tr>
            <tr><td><strong>🔄 Sinkronisasi</strong></td><td>Otomatis saat aplikasi dijalankan + scheduler harian</td></tr>
            <tr><td><strong>💾 Penyimpanan</strong></td><td>SQLite database (migrasi otomatis dari CSV)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🌽 10 Komoditas Strategis yang Dipantau")
    commodities = [
        ("🍚", "Beras", "Makanan pokok utama Indonesia"),
        ("🍗", "Daging Ayam", "Sumber protein utama"),
        ("🥩", "Daging Sapi", "Komoditas pangan premium"),
        ("🥚", "Telur Ayam", "Sumber protein terjangkau"),
        ("🧅", "Bawang Merah", "Bumbu dapur utama, volatilitas tinggi"),
        ("🧄", "Bawang Putih", "Bumbu dapur utama, mayoritas impor"),
        ("🌶️", "Cabai Merah", "Bumbu utama, sangat volatile"),
        ("🌶️", "Cabai Rawit", "Bumbu utama, sering mengalami spike"),
        ("🫗", "Minyak Goreng", "Kebutuhan rumah tangga esensial"),
        ("🍬", "Gula Pasir", "Kebutuhan industri dan rumah tangga"),
    ]
    
    cols = st.columns(2)
    for i, (icon, name, desc) in enumerate(commodities):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
                padding: 10px 15px;
                margin: 4px 0;
            ">
                {icon} <strong>{name}</strong> — <span style="color: #aaa;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 🔧 Cara Pengambilan Data")
    st.markdown("""
    1. **Scraper** (`pihps_scraper.py`) mengambil data JSON dari endpoint API PIHPS BI
    2. **Harmonizer** (`pihps_harmonizer.py`) membersihkan data: interpolasi hari libur, normalisasi nama komoditas
    3. **Database** (`data/database.py`) menyimpan ke SQLite dengan indexing untuk query cepat
    4. **Auto-sync** berjalan otomatis setiap kali aplikasi dijalankan untuk mengambil data terbaru
    """)

with ds_tab2:
    st.markdown("### 🌦️ Data Cuaca — Open-Meteo API")
    
    st.markdown("""
    <div style="
        background: rgba(0,204,150,0.08);
        border: 1px solid rgba(0,204,150,0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    ">
        <table style="width: 100%; color: #ddd;">
            <tr><td style="width: 160px;"><strong>📡 Sumber</strong></td><td>Open-Meteo — Free Weather API</td></tr>
            <tr><td><strong>🌐 Website</strong></td><td><a href="https://open-meteo.com" target="_blank">https://open-meteo.com</a></td></tr>
            <tr><td><strong>📜 Lisensi</strong></td><td>CC BY 4.0 (gratis untuk non-komersial, atribusi wajib)</td></tr>
            <tr><td><strong>🔑 API Key</strong></td><td>❌ Tidak diperlukan</td></tr>
            <tr><td><strong>📅 Rentang Data</strong></td><td>1940 — Sekarang (data historis harian)</td></tr>
            <tr><td><strong>🗺️ Cakupan</strong></td><td>Global (termasuk seluruh Indonesia)</td></tr>
            <tr><td><strong>📊 Parameter</strong></td><td>Curah hujan harian (mm), Suhu rata-rata (°C)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📍 Koordinat Per Provinsi")
    st.markdown("""
    Data cuaca diambil berdasarkan **koordinat ibukota provinsi**. 
    Setiap provinsi memiliki titik referensi yang sudah di-mapping di sistem.
    """)

    # Show province coordinates table
    try:
        from data.weather_client import PROVINCE_COORDS
        coords_data = [
            {"Provinsi": prov, "Latitude": f"{lat:.2f}", "Longitude": f"{lon:.2f}"}
            for prov, (lat, lon) in sorted(PROVINCE_COORDS.items())
        ]
        st.dataframe(pd.DataFrame(coords_data), use_container_width=True, height=300)
    except ImportError:
        st.info("Data koordinat tidak tersedia.")

    st.markdown("#### 🔗 Mengapa Open-Meteo?")
    st.markdown("""
    | Kriteria | Open-Meteo | BMKG |
    |----------|-----------|------|
    | **Gratis** | ✅ Ya | ⚠️ Perlu registrasi |
    | **API Key** | ❌ Tidak perlu | ✅ Perlu |
    | **Coverage** | 🌍 Global | 🇮🇩 Indonesia only |
    | **Data Historis** | 📅 Sejak 1940 | 📅 Terbatas |
    | **Format** | JSON (REST API) | Varies |
    | **Uptime** | 99.9% | N/A |
    """)

    # Live test
    st.markdown("#### 🧪 Test Data Cuaca Live")
    test_prov = st.selectbox("Pilih Provinsi untuk Test", list(PROVINCE_COORDS.keys()), index=11, key="about_prov")
    
    if st.button("🔍 Ambil Data Cuaca Sekarang", key="test_weather"):
        with st.spinner(f"Mengambil data cuaca {test_prov} dari Open-Meteo..."):
            try:
                from data.weather_client import WeatherClient
                client = WeatherClient(use_live=True)
                dates = pd.date_range(
                    pd.Timestamp.now() - pd.Timedelta(days=7),
                    pd.Timestamp.now() - pd.Timedelta(days=1)
                )
                features = client.get_weather_features(test_prov, dates)
                
                st.success(f"✅ Data berhasil diambil dari Open-Meteo untuk {test_prov}!")
                
                display_df = features.copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                display_df.columns = ['Curah Hujan (mm)', 'Suhu (°C)', 'ENSO Index', 'Musim Hujan']
                st.dataframe(display_df, use_container_width=True)
                
                source = client.get_data_source_info()
                st.caption(f"📡 Weather: {source['weather']['provider']} ({source['weather']['license']}) | ENSO: {source['enso']['provider']}")
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")

with ds_tab3:
    st.markdown("### 🌊 Data ENSO (El Niño / La Niña) — NOAA")
    
    st.markdown("""
    <div style="
        background: rgba(255,165,0,0.08);
        border: 1px solid rgba(255,165,0,0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    ">
        <table style="width: 100%; color: #ddd;">
            <tr><td style="width: 160px;"><strong>📡 Sumber</strong></td><td>NOAA Climate Prediction Center (CPC)</td></tr>
            <tr><td><strong>🌐 Website</strong></td><td><a href="https://www.cpc.ncep.noaa.gov" target="_blank">https://www.cpc.ncep.noaa.gov</a></td></tr>
            <tr><td><strong>📜 Lisensi</strong></td><td>Public Domain (data milik pemerintah AS)</td></tr>
            <tr><td><strong>📊 Indeks</strong></td><td>Oceanic Niño Index (ONI)</td></tr>
            <tr><td><strong>📅 Rentang</strong></td><td>1950 — Sekarang (update bulanan)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 📖 Apa itu ENSO?
    
    **ENSO** (El Niño–Southern Oscillation) adalah fenomena iklim global yang sangat 
    berpengaruh terhadap cuaca Indonesia:
    
    | Kondisi | ONI Value | Dampak di Indonesia |
    |---------|-----------|---------------------|
    | **El Niño** 🌡️ | > +0.5 | Kemarau panjang → gagal panen → harga naik |
    | **Netral** ⚖️ | -0.5 s/d +0.5 | Cuaca normal |
    | **La Niña** 🌧️ | < -0.5 | Curah hujan berlebih → banjir → distribusi terganggu |
    
    Data ENSO digunakan sebagai **external regressor** pada model Prophet untuk 
    meningkatkan akurasi prediksi pada periode El Niño/La Niña.
    """)

# =====================================================================
# 3. MODEL AI
# =====================================================================
st.markdown("---")
st.markdown("## 🧠 Model AI")

m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["📈 Prophet", "🔄 BiLSTM", "🔮 TFT", "🎯 Ensemble"])

with m_tab1:
    st.markdown("### 📈 Prophet (Facebook/Meta)")
    st.markdown("""
    **Prophet** adalah model time-series dari Meta yang dirancang untuk data bisnis 
    dengan pola musiman yang kuat.
    
    | Aspek | Detail |
    |-------|--------|
    | **Tipe** | Additive regression model |
    | **Komponen** | Trend + Yearly Seasonality + Weekly Seasonality |
    | **Confidence Interval** | 90% (configurable) |
    | **External Regressors** | Curah hujan, suhu, ENSO index (opsional) |
    | **Kelebihan** | Robust terhadap missing data, mudah di-tune, interpretable |
    | **Keterbatasan** | Kurang baik untuk short-term spikes |
    | **Digunakan untuk** | Tren jangka menengah-panjang (14-120 hari) |
    """)

with m_tab2:
    st.markdown("### 🔄 Bidirectional LSTM (PyTorch)")
    st.markdown("""
    **Bidirectional Long Short-Term Memory (BiLSTM)** adalah arsitektur deep learning 
    yang memproses data time-series dari dua arah (past→future dan future→past).
    
    | Aspek | Detail |
    |-------|--------|
    | **Arsitektur** | 2-layer Bidirectional LSTM + Dropout + Linear |
    | **Hidden Size** | 128 units per direction (256 total) |
    | **Sequence Length** | 30 hari (lookback window) |
    | **Output** | Single-step atau Multi-step (recursive) |
    | **Uncertainty** | MC Dropout (50 forward passes untuk confidence interval) |
    | **Kelebihan** | Menangkap pola temporal kompleks, baik untuk short-term |
    | **Keterbatasan** | Error accumulation pada multi-step panjang |
    | **Digunakan untuk** | Prediksi jangka pendek (1-14 hari) |
    
    #### Arsitektur:
    ```
    Input (30 days) → BiLSTM Layer 1 (128×2) → Dropout (0.2)
                    → BiLSTM Layer 2 (128×2) → Dropout (0.2)
                    → Linear (256 → 1) → Price Prediction
    ```
    """)

with m_tab3:
    st.markdown("### 🔮 Temporal Fusion Transformer (TFT)")
    st.markdown("""
    **TFT** adalah model state-of-the-art dari Google untuk multi-horizon time-series 
    forecasting yang menggabungkan LSTM dan attention mechanism.
    
    | Aspek | Detail |
    |-------|--------|
    | **Tipe** | Attention-based encoder-decoder |
    | **Library** | pytorch-forecasting (opsional) |
    | **Input** | Static (provinsi, komoditas) + Time-varying (harga, cuaca) |
    | **Output** | Multi-horizon simultaneous (1-30 hari sekaligus) |
    | **Kelebihan** | Interpretable (variable importance), multi-horizon |
    | **Status** | ⚠️ Opsional (dependency besar: pytorch-forecasting + pytorch-lightning) |
    """)

    st.info("ℹ️ TFT memerlukan library tambahan. Install dengan: `pip install pytorch-forecasting pytorch-lightning`")

with m_tab4:
    st.markdown("### 🎯 Smart Ensemble")
    st.markdown("""
    **Smart Ensemble** menggabungkan prediksi dari ketiga model dengan bobot adaptif:
    
    | Horizon | Prophet | LSTM | TFT |
    |---------|---------|------|-----|
    | **1-7 hari** (short-term) | 25% | 50% | 25% |
    | **8-30 hari** (medium-term) | 35% | 30% | 35% |
    | **31+ hari** (long-term) | 45% | 10% | 45% |
    
    #### Fitur Adaptif:
    - **Dynamic Weight**: Bobot otomatis diperbarui berdasarkan MAPE terbaru tiap model
    - **Fallback Cascade**: Jika model tidak tersedia → gunakan yang tersedia saja
    - **Confidence Interval**: Gabungan terluas dari semua model (union bound)
    """)

# =====================================================================
# 4. EWS ENGINE
# =====================================================================
st.markdown("---")
st.markdown("## ⚠️ Engine Peringatan Dini (EWS v2)")

st.markdown("""
Sistem EWS v2 menggunakan **5 faktor** untuk menghitung skor risiko komposit (0-100):
""")

factors_data = [
    {
        "Faktor": "📈 Perubahan Harga",
        "Bobot": "30%",
        "Deskripsi": "Persentase perubahan harga prediksi vs aktual. Kenaikan > 20% = skor tinggi.",
    },
    {
        "Faktor": "📊 Volatilitas Pasar",
        "Bobot": "20%",
        "Deskripsi": "Standar deviasi rolling 30 hari dari perubahan harga harian.",
    },
    {
        "Faktor": "📅 Anomali Musiman",
        "Bobot": "20%",
        "Deskripsi": "Z-score prediksi vs rata-rata historis bulan yang sama. Deteksi penyimpangan dari pola normal.",
    },
    {
        "Faktor": "🗺️ Anomali Regional",
        "Bobot": "15%",
        "Deskripsi": "Seberapa jauh harga provinsi menyimpang dari rata-rata nasional. Deteksi masalah distribusi.",
    },
    {
        "Faktor": "🚀 Kecepatan Perubahan",
        "Bobot": "15%",
        "Deskripsi": "Akselerasi kenaikan harga (7d vs 14d vs 30d). Semakin cepat = semakin bahaya.",
    },
]

st.dataframe(pd.DataFrame(factors_data), use_container_width=True, hide_index=True)

st.markdown("""
#### 🚦 Level Peringatan

| Skor | Level | Warna | Tindakan |
|------|-------|-------|----------|
| **70-100** | 🔴 DANGER | Merah | Intervensi segera diperlukan |
| **45-69** | 🟠 ALERT | Oranye | Monitoring intensif, siapkan contingency |
| **25-44** | 🟡 WATCH | Kuning | Pantau perkembangan 7 hari ke depan |
| **0-24** | 🟢 NORMAL | Hijau | Tidak perlu tindakan khusus |
""")

# =====================================================================
# 5. ARSITEKTUR SISTEM
# =====================================================================
st.markdown("---")
st.markdown("## 🏗️ Arsitektur Sistem")

st.markdown("""
```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ PIHPS BI │  │Open-Meteo│  │ NOAA CPC │              │
│  │(Harga)   │  │(Cuaca)   │  │(ENSO)    │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│       ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────┐            │
│  │         SQLite Database Layer           │            │
│  │   (data/database.py + scheduler.py)     │            │
│  └──────────────────┬──────────────────────┘            │
│                     │                                    │
│       ┌─────────────┼─────────────┐                    │
│       ▼             ▼             ▼                    │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐               │
│  │ Prophet │  │ BiLSTM   │  │  TFT    │               │
│  │ +Weather│  │ +MC Drop │  │ +Attn   │               │
│  └────┬────┘  └────┬─────┘  └────┬────┘               │
│       │             │             │                     │
│       └─────────┬───┘─────────────┘                     │
│                 ▼                                        │
│  ┌──────────────────────────────────┐                   │
│  │        Smart Ensemble            │                   │
│  │  (Adaptive Weight + Fallback)    │                   │
│  └──────────────┬───────────────────┘                   │
│                 │                                        │
│       ┌─────────┼─────────────┐                        │
│       ▼                       ▼                        │
│  ┌──────────┐          ┌──────────────┐                │
│  │ EWS v2   │          │ Supply Risk  │                │
│  │(5-Factor)│          │ (4-Factor)   │                │
│  └────┬─────┘          └──────┬───────┘                │
│       │                       │                         │
│       └───────────┬───────────┘                         │
│                   ▼                                      │
│  ┌────────────────────────────────────────┐             │
│  │           OUTPUT LAYER                 │             │
│  │  ┌──────────┐  ┌────────┐  ┌───────┐ │             │
│  │  │Streamlit │  │FastAPI │  │ PDF/  │ │             │
│  │  │Dashboard │  │  REST  │  │ Excel │ │             │
│  │  └──────────┘  └────────┘  └───────┘ │             │
│  └────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```
""")

# =====================================================================
# 6. API
# =====================================================================
st.markdown("---")
st.markdown("## 🔌 REST API")

st.markdown("""
Sistem menyediakan REST API (FastAPI) yang bisa diakses oleh aplikasi lain.

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/forecast` | GET | Prediksi harga komoditas |
| `/api/ews/status` | GET | Status peringatan dini semua komoditas |
| `/api/models/compare` | GET | Perbandingan performa model |
| `/api/data/prices` | GET | Data harga historis (filterable) |
| `/api/data/commodities` | GET | Daftar komoditas |
| `/api/data/provinces` | GET | Daftar provinsi |
| `/api/data/stats` | GET | Statistik database |
| `/api/data/supply-risk` | GET | Skor risiko pasokan |
| `/api/data/sync` | POST | Trigger sinkronisasi data PIHPS |
| `/docs` | GET | Swagger UI (auto-generated documentation) |

**Base URL**: `http://localhost:8000` (saat dijalankan dengan `python run.py`)
""")

# =====================================================================
# 7. CREDITS
# =====================================================================
st.markdown("---")
st.markdown("## 📜 Credits & Atribusi")

st.markdown("""
<div style="
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
">

| Komponen | Provider | Lisensi |
|----------|----------|---------|
| Data Harga Pangan | **Bank Indonesia** (PIHPS) | Data publik |
| Data Cuaca | **Open-Meteo** | CC BY 4.0 |
| Data ENSO | **NOAA Climate Prediction Center** | Public Domain |
| Prophet Model | **Meta (Facebook)** | MIT License |
| PyTorch | **Meta AI Research** | BSD License |
| Streamlit | **Snowflake** | Apache 2.0 |
| FastAPI | **Sebastián Ramírez** | MIT License |
| GeoJSON Indonesia | **superpikar/indonesia-geojson** | Open Data |

</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="text-align: center; opacity: 0.5; font-size: 0.85rem; padding: 20px 0;">
    <div><strong>Agri-AI Early Warning System v2.0</strong></div>
    <div>© 2026 Fahmi Prasanda</div>
    <div style="margin-top: 8px;">
        Engine: Prophet + BiLSTM + TFT | 
        Weather: Open-Meteo (CC BY 4.0) | 
        ENSO: NOAA CPC
    </div>
</div>
""", unsafe_allow_html=True)
