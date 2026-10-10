"""
Page 5: About: Detailed explanation of the system, data sources, and methodology.
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Tentang | Agri-AI EWS", page_icon="🌾", layout="wide")

# --- Theme ---
from theme import inject_theme_css, render_theme_toggle, render_sidebar_brand, render_footer, status_chip
inject_theme_css()

render_sidebar_brand("Tentang sistem")
render_theme_toggle()

# --- Header ---
st.title("Tentang Agri-AI EWS")
st.markdown("Dokumentasi sumber data, model, dan logika peringatan dini harga pangan.")

st.divider()

# =====================================================================
# 1. OVERVIEW
# =====================================================================
st.header("Apa itu Agri-AI EWS?")
st.markdown("""
**Agri-AI Early Warning System (EWS)** memantau, memprakirakan, dan memberi peringatan dini
atas perubahan harga **10 komoditas pangan strategis** di **34 provinsi** yang dicakup PIHPS.

Sistem ini menggabungkan:
- **Tiga model prakiraan**: Prophet, Bidirectional LSTM, dan Temporal Fusion Transformer
- **Data harga harian** dari PIHPS Bank Indonesia
- **Data cuaca dan iklim** dari Open-Meteo dan NOAA
- **Skor peringatan dini multifaktor** untuk mendeteksi risiko lonjakan harga
""")

# =====================================================================
# 2. DATA SOURCES
# =====================================================================
st.divider()
st.header("Sumber data")

ds_tab1, ds_tab2, ds_tab3 = st.tabs(["Harga pangan", "Cuaca", "ENSO"])

with ds_tab1:
    st.subheader("Harga pangan: PIHPS Bank Indonesia")

    st.markdown("""
    <div class="panel">
        <table class="theme-table">
            <tr><td>Sumber</td><td>Pusat Informasi Harga Pangan Strategis Nasional (PIHPS), Bank Indonesia</td></tr>
            <tr><td>Situs</td><td><a href="https://www.bi.go.id/hargapangan" target="_blank" rel="noopener">bi.go.id/hargapangan</a></td></tr>
            <tr><td>Rentang</td><td>Januari 2021 sampai sekarang (diperbarui setiap hari kerja)</td></tr>
            <tr><td>Cakupan</td><td>34 provinsi</td></tr>
            <tr><td>Sinkronisasi</td><td>Otomatis saat aplikasi dijalankan, lalu terjadwal setiap hari</td></tr>
            <tr><td>Penyimpanan</td><td>Database SQLite (migrasi otomatis dari CSV)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 10 komoditas yang dipantau")
    commodities = [
        ("Beras", "makanan pokok utama"),
        ("Daging Ayam", "sumber protein utama"),
        ("Daging Sapi", "protein dengan harga tertinggi di daftar ini"),
        ("Telur Ayam", "sumber protein terjangkau"),
        ("Bawang Merah", "bumbu utama, volatilitas tinggi"),
        ("Bawang Putih", "bumbu utama, sebagian besar impor"),
        ("Cabai Merah", "bumbu utama, sangat volatil"),
        ("Cabai Rawit", "bumbu utama, sering melonjak"),
        ("Minyak Goreng", "kebutuhan rumah tangga"),
        ("Gula Pasir", "kebutuhan industri dan rumah tangga"),
    ]
    cols = st.columns(2)
    for i, (name, desc) in enumerate(commodities):
        cols[i % 2].markdown(f"- **{name}**: {desc}")

    st.markdown("#### Cara pengambilan data")
    st.markdown("""
    1. **Scraper** (`pihps_scraper.py`) mengambil data JSON dari endpoint PIHPS BI.
    2. **Harmonizer** (`pihps_harmonizer.py`) membersihkan data: interpolasi hari libur dan normalisasi nama komoditas.
    3. **Database** (`data/database.py`) menyimpan ke SQLite dengan indeks untuk kueri cepat.
    4. **Sinkronisasi otomatis** mengambil data terbaru setiap kali aplikasi dijalankan.
    """)

with ds_tab2:
    st.subheader("Cuaca: Open-Meteo")

    st.markdown("""
    <div class="panel">
        <table class="theme-table">
            <tr><td>Sumber</td><td>Open-Meteo Historical Weather API</td></tr>
            <tr><td>Situs</td><td><a href="https://open-meteo.com" target="_blank" rel="noopener">open-meteo.com</a></td></tr>
            <tr><td>Lisensi</td><td>CC BY 4.0 (gratis untuk non-komersial, wajib atribusi)</td></tr>
            <tr><td>API key</td><td>Tidak diperlukan</td></tr>
            <tr><td>Rentang data</td><td>1940 sampai sekarang (data historis harian)</td></tr>
            <tr><td>Cakupan</td><td>Global, termasuk seluruh Indonesia</td></tr>
            <tr><td>Parameter</td><td>Curah hujan harian (mm), suhu rata-rata (°C)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Koordinat per provinsi")
    st.markdown("Data cuaca diambil di **koordinat ibu kota provinsi**. Setiap provinsi punya satu titik referensi.")

    PROVINCE_COORDS = {}
    try:
        from data.weather_client import PROVINCE_COORDS
        coords_data = [
            {"Provinsi": prov, "Lintang": f"{lat:.2f}", "Bujur": f"{lon:.2f}"}
            for prov, (lat, lon) in sorted(PROVINCE_COORDS.items())
        ]
        st.dataframe(pd.DataFrame(coords_data), use_container_width=True, height=300, hide_index=True)
    except ImportError:
        st.info("Tabel koordinat tidak bisa dimuat karena modul data.weather_client tidak ditemukan.")

    st.markdown("#### Alasan memakai Open-Meteo")
    st.markdown("""
    - Tidak perlu API key, jadi sinkronisasi bisa berjalan otomatis tanpa kredensial.
    - Menyediakan arsip harian sejak 1940, cukup panjang untuk seluruh periode data harga.
    - Satu format JSON yang sama untuk semua provinsi.
    """)

    # Live test
    if PROVINCE_COORDS:
        st.markdown("#### Coba ambil data cuaca")
        test_prov = st.selectbox("Provinsi", list(PROVINCE_COORDS.keys()), index=min(11, len(PROVINCE_COORDS) - 1),
                                 key="about_prov")

        if st.button("Ambil data cuaca 7 hari terakhir", key="test_weather"):
            with st.spinner(f"Mengambil data cuaca {test_prov} dari Open-Meteo..."):
                try:
                    from data.weather_client import WeatherClient
                    client = WeatherClient(use_live=True)
                    dates = pd.date_range(
                        pd.Timestamp.now() - pd.Timedelta(days=7),
                        pd.Timestamp.now() - pd.Timedelta(days=1)
                    )
                    features = client.get_weather_features(test_prov, dates)

                    st.success(f"Data cuaca {test_prov} berhasil diambil dari Open-Meteo.")

                    display_df = features.copy()
                    display_df.index = display_df.index.strftime('%Y-%m-%d')
                    display_df.columns = ['Curah hujan (mm)', 'Suhu (°C)', 'Indeks ENSO', 'Musim hujan']
                    st.dataframe(display_df, use_container_width=True)

                    source = client.get_data_source_info()
                    st.caption(f"Cuaca: {source['weather']['provider']} ({source['weather']['license']}). "
                               f"ENSO: {source['enso']['provider']}.")
                except Exception as e:
                    st.error("Data cuaca gagal diambil. Periksa koneksi internet lalu coba lagi. "
                             f"(Detail teknis: {type(e).__name__}: {e})")

with ds_tab3:
    st.subheader("ENSO (El Niño / La Niña): NOAA")

    st.markdown("""
    <div class="panel">
        <table class="theme-table">
            <tr><td>Sumber</td><td>NOAA Climate Prediction Center (CPC)</td></tr>
            <tr><td>Situs</td><td><a href="https://www.cpc.ncep.noaa.gov" target="_blank" rel="noopener">cpc.ncep.noaa.gov</a></td></tr>
            <tr><td>Lisensi</td><td>Domain publik (data pemerintah AS)</td></tr>
            <tr><td>Indeks</td><td>Oceanic Niño Index (ONI)</td></tr>
            <tr><td>Rentang</td><td>1950 sampai sekarang (diperbarui bulanan)</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### Apa itu ENSO?

    **ENSO** (El Niño-Southern Oscillation) adalah fenomena iklim global yang berpengaruh
    besar terhadap cuaca Indonesia:

    | Kondisi | Nilai ONI | Dampak di Indonesia |
    |---------|-----------|---------------------|
    | **El Niño** | > +0,5 | Kemarau panjang, risiko gagal panen, harga cenderung naik |
    | **Netral** | -0,5 s.d. +0,5 | Cuaca normal |
    | **La Niña** | < -0,5 | Curah hujan berlebih, risiko banjir, distribusi terganggu |

    Data ENSO dipakai sebagai **regresor eksternal** pada model Prophet untuk
    menangkap pengaruh periode El Niño dan La Niña.
    """)

# =====================================================================
# 3. MODEL
# =====================================================================
st.divider()
st.header("Model prakiraan")

m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs(["Prophet", "BiLSTM", "TFT", "Ensemble"])

with m_tab1:
    st.subheader("Prophet (Meta)")
    st.markdown("""
    **Prophet** adalah model deret waktu dari Meta untuk data dengan pola musiman yang kuat
    (Taylor & Letham, 2018).

    | Aspek | Detail |
    |-------|--------|
    | **Tipe** | Model regresi aditif |
    | **Komponen** | Tren + musiman tahunan + musiman mingguan |
    | **Interval** | 90% (bisa diatur) |
    | **Regresor eksternal** | Curah hujan, indeks ENSO, musim hujan/kemarau (opsional) |
    | **Kelebihan** | Tahan terhadap data hilang, mudah disetel, mudah ditafsirkan |
    | **Keterbatasan** | Kurang menangkap lonjakan jangka pendek |
    | **Dipakai untuk** | Tren jangka menengah-panjang (14-120 hari) |
    """)

with m_tab2:
    st.subheader("Bidirectional LSTM (PyTorch)")
    st.markdown("""
    **Bidirectional Long Short-Term Memory (BiLSTM)** adalah arsitektur deep learning
    yang memproses urutan data dari dua arah.

    | Aspek | Detail |
    |-------|--------|
    | **Arsitektur** | 2 lapis Bidirectional LSTM + Dropout + Linear |
    | **Hidden size** | 128 unit per arah (256 total) |
    | **Panjang sekuens** | 30 hari |
    | **Input** | Univariat: 30 harga harian terakhir, diskalakan ke [-1, 1] |
    | **Pelatihan** | Mini-batch 64, Adam (lr 0,001), maks. 50 epoch + early stopping (patience 5) |
    | **Output** | Satu langkah atau banyak langkah (rekursif) |
    | **Ketidakpastian** | MC Dropout (50 forward pass untuk interval) |
    | **Kelebihan** | Menangkap pola temporal kompleks, baik untuk jangka pendek |
    | **Keterbatasan** | Galat menumpuk pada prakiraan rekursif yang panjang |
    | **Dipakai untuk** | Prediksi jangka pendek (1-14 hari) |

    #### Arsitektur
    ```
    Input (30 hari) -> BiLSTM lapis 1 (128x2) -> Dropout (0.2)
                    -> BiLSTM lapis 2 (128x2) -> Dropout (0.2)
                    -> Linear (256 -> 1) -> Prediksi harga
    ```
    """)

with m_tab3:
    st.subheader("Temporal Fusion Transformer (TFT)")
    st.markdown("""
    **TFT** adalah model multi-horizon berbasis attention yang menggabungkan LSTM dan
    mekanisme attention (Lim et al., 2021).

    | Aspek | Detail |
    |-------|--------|
    | **Tipe** | Encoder-decoder berbasis attention |
    | **Library** | pytorch-forecasting (opsional) |
    | **Input** | Statis (provinsi, komoditas) + berubah terhadap waktu (harga, cuaca) |
    | **Output** | Multi-horizon sekaligus (1-30 hari) |
    | **Kelebihan** | Menyediakan variable importance, multi-horizon |
    | **Status** | Opsional (dependensi besar: pytorch-forecasting + pytorch-lightning) |
    """)

    st.info("TFT memerlukan library tambahan. Pasang dengan: `pip install pytorch-forecasting pytorch-lightning`")

with m_tab4:
    st.subheader("Ensemble")
    st.markdown("""
    **Ensemble** menggabungkan prediksi Prophet, BiLSTM, dan TFT dengan bobot yang
    **dicari dari data**, bukan ditetapkan manual:

    | Tahap | Keterangan |
    |-------|------------|
    | **Periode validasi** | 20% terakhir data latih (sebelum periode uji) |
    | **Pencarian bobot** | Grid search kelipatan 5% (jumlah bobot = 100%), meminimalkan MAPE validasi |
    | **Adaptif** | Bobot dihitung ulang setiap kali model dilatih ulang dengan data terbaru |
    | **Fallback** | Jika sebaran prediksi antarmodel (koefisien variasi) > 15%, dipakai model dengan MAPE validasi terbaik |
    | **Interval** | Gabungan terluas dari interval model penyusun |

    Bobot untuk komoditas dan provinsi terpilih ditampilkan di Dashboard, dan di
    Laboratorium Model setelah perbandingan model dijalankan.
    """)

# =====================================================================
# 4. EWS ENGINE
# =====================================================================
st.divider()
st.header("Logika peringatan dini (EWS v2)")

st.markdown("EWS v2 memakai **5 faktor** untuk menghitung skor risiko komposit (0-100):")

factors_data = [
    {"Faktor": "Perubahan harga", "Bobot": "30%",
     "Deskripsi": "Persentase perubahan harga prediksi terhadap aktual. Kenaikan > 20% = skor tinggi."},
    {"Faktor": "Volatilitas pasar", "Bobot": "20%",
     "Deskripsi": "Simpangan baku bergulir 30 hari dari perubahan harga harian."},
    {"Faktor": "Anomali musiman", "Bobot": "20%",
     "Deskripsi": "Z-score prediksi terhadap rata-rata historis bulan yang sama."},
    {"Faktor": "Anomali regional", "Bobot": "15%",
     "Deskripsi": "Seberapa jauh harga provinsi menyimpang dari rata-rata nasional."},
    {"Faktor": "Kecepatan perubahan", "Bobot": "15%",
     "Deskripsi": "Percepatan kenaikan harga (7 vs 14 vs 30 hari). Makin cepat, makin berisiko."},
]
st.dataframe(pd.DataFrame(factors_data), use_container_width=True, hide_index=True)

st.markdown("#### Level peringatan")
levels = [
    ("70-100", "Bahaya", "danger", "Intervensi segera diperlukan"),
    ("45-69", "Waspada", "alert", "Pemantauan intensif, siapkan rencana kontingensi"),
    ("25-44", "Perhatian", "watch", "Pantau perkembangan 7 hari ke depan"),
    ("0-24", "Normal", "normal", "Tidak perlu tindakan khusus"),
]
rows = "".join(f"<tr><td>{status_chip(label, status)}</td><td>Skor {score}. {action}.</td></tr>"
               for score, label, status, action in levels)
st.markdown(f'<div class="panel"><table class="theme-table">{rows}</table></div>', unsafe_allow_html=True)

# =====================================================================
# 5. ARSITEKTUR SISTEM
# =====================================================================
st.divider()
st.header("Arsitektur sistem")

st.markdown("""
```
SUMBER DATA      PIHPS BI (harga)   Open-Meteo (cuaca)   NOAA CPC (ENSO)
                        |                  |                   |
                        +------------------+-------------------+
                                           |
PENYIMPANAN               SQLite (data/database.py + scheduler.py)
                                           |
                        +------------------+-------------------+
                        |                  |                   |
MODEL              Prophet + cuaca   BiLSTM + MC Dropout   TFT + attention
                        |                  |                   |
                        +------------------+-------------------+
                                           |
                           Ensemble (bobot adaptif + fallback)
                                           |
                        +------------------+-------------------+
                        |                                      |
ANALISIS          EWS v2 (5 faktor)                Risiko pasokan (4 faktor)
                        |                                      |
                        +------------------+-------------------+
                                           |
KELUARAN          Streamlit dashboard  |  FastAPI REST  |  PDF / Excel
```
""")

# =====================================================================
# 6. API
# =====================================================================
st.divider()
st.header("REST API")

st.markdown("""
Sistem menyediakan REST API (FastAPI) yang bisa diakses aplikasi lain.

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/forecast` | GET | Prediksi harga komoditas |
| `/api/ews/status` | GET | Status peringatan dini semua komoditas |
| `/api/models/compare` | GET | Perbandingan performa model |
| `/api/data/prices` | GET | Data harga historis (bisa difilter) |
| `/api/data/latest` | GET | Harga terbaru per provinsi dan komoditas |
| `/api/data/commodities` | GET | Daftar komoditas |
| `/api/data/provinces` | GET | Daftar provinsi |
| `/api/data/stats` | GET | Statistik database |
| `/api/data/supply-risk` | GET | Skor risiko pasokan |
| `/api/data/sync` | POST | Memicu sinkronisasi data PIHPS |
| `/health` | GET | Status layanan dan database |
| `/docs` | GET | Swagger UI (dokumentasi otomatis) |

**Base URL**: `http://localhost:8000` (saat dijalankan dengan `python run.py`)
""")

# =====================================================================
# 7. CREDITS
# =====================================================================
st.divider()
st.header("Kredit dan atribusi")

st.markdown("""
| Komponen | Penyedia | Lisensi |
|----------|----------|---------|
| Data harga pangan | **Bank Indonesia** (PIHPS) | Data publik |
| Data cuaca | **Open-Meteo** | CC BY 4.0 |
| Data ENSO | **NOAA Climate Prediction Center** | Domain publik |
| Prophet | **Meta** | MIT License |
| PyTorch | **Meta AI Research** | BSD License |
| Streamlit | **Snowflake** | Apache 2.0 |
| FastAPI | **Sebastián Ramírez** | MIT License |
| GeoJSON Indonesia | **superpikar/indonesia-geojson** | Open Data |
| Font IBM Plex Sans | **IBM** (lewat Google Fonts) | SIL Open Font License |
""")

st.divider()
render_footer("Agri-AI Early Warning System v2.0", "Cuaca: Open-Meteo (CC BY 4.0), ENSO: NOAA CPC",
              "© 2026 Fahmi Prasanda")
