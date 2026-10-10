"""
Page 2: Regional Analysis: Indonesia Choropleth Map & Provincial Drill-down.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_store

st.set_page_config(page_title="Analisis Regional | Agri-AI EWS", page_icon="🌾", layout="wide")

# --- Theme ---
from theme import inject_theme_css, render_theme_toggle, render_sidebar_brand, apply_theme_to_plotly, CHART, STATUS
inject_theme_css()

# Province name mapping: our data names → GeoJSON names
PROVINCE_TO_GEOJSON = {
    "Aceh": "DI. ACEH", "Bali": "BALI", "Banten": "PROBANTEN",
    "Bengkulu": "BENGKULU", "DI Yogyakarta": "DAERAH ISTIMEWA YOGYAKARTA",
    "DKI Jakarta": "DKI JAKARTA", "Gorontalo": "GORONTALO",
    "Jambi": "JAMBI", "Jawa Barat": "JAWA BARAT",
    "Jawa Tengah": "JAWA TENGAH", "Jawa Timur": "JAWA TIMUR",
    "Kalimantan Barat": "KALIMANTAN BARAT", "Kalimantan Selatan": "KALIMANTAN SELATAN",
    "Kalimantan Tengah": "KALIMANTAN TENGAH", "Kalimantan Timur": "KALIMANTAN TIMUR",
    "Kalimantan Utara": "KALIMANTAN TIMUR",  # Merged in old GeoJSON
    "Kepulauan Bangka Belitung": "BANGKA BELITUNG",
    "Kepulauan Riau": "RIAU",  # Merged in old GeoJSON
    "Lampung": "LAMPUNG", "Maluku": "MALUKU", "Maluku Utara": "MALUKU UTARA",
    "Nusa Tenggara Barat": "NUSATENGGARA BARAT", "Nusa Tenggara Timur": "NUSA TENGGARA TIMUR",
    "Papua": "IRIAN JAYA TIMUR", "Papua Barat": "IRIAN JAYA BARAT",
    "Riau": "RIAU", "Sulawesi Barat": "SULAWESI SELATAN",  # Merged
    "Sulawesi Selatan": "SULAWESI SELATAN", "Sulawesi Tengah": "SULAWESI TENGAH",
    "Sulawesi Tenggara": "SULAWESI TENGGARA", "Sulawesi Utara": "SULAWESI UTARA",
    "Sumatera Barat": "SUMATERA BARAT", "Sumatera Selatan": "SUMATERA SELATAN",
    "Sumatera Utara": "SUMATERA UTARA",
}

GEOJSON_TO_PROVINCE = {v: k for k, v in PROVINCE_TO_GEOJSON.items()}

@st.cache_data
def load_data():
    store = get_store()
    df = store.load_all()
    if df.empty:
        csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "food_prices_real.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df

@st.cache_data
def load_geojson():
    geo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "indonesia.geojson")
    if os.path.exists(geo_path):
        with open(geo_path) as f:
            return json.load(f)
    return None

df = load_data()
geojson = load_geojson()

if df.empty:
    st.error("Belum ada data harga. Buka halaman utama untuk menjalankan sinkronisasi PIHPS, "
             "atau letakkan food_prices_real.csv di folder proyek lalu muat ulang halaman.")
    st.stop()

# --- Sidebar ---
render_sidebar_brand("Analisis regional")
render_theme_toggle()
st.sidebar.divider()
selected_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), index=0)

# --- Main Content ---
st.title("Analisis Regional")
st.markdown(f"Harga **{selected_commodity}** terakhir di seluruh provinsi")

# --- Choropleth Map ---
if geojson is not None:
    latest_prices = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
    latest_prices['geojson_name'] = latest_prices['province'].map(PROVINCE_TO_GEOJSON)
    latest_prices = latest_prices.dropna(subset=['geojson_name'])

    # Aggregate duplicates (provinces that map to same geojson region)
    latest_prices = latest_prices.groupby('geojson_name').agg({
        'province': 'first', 'price': 'mean', 'date': 'first'
    }).reset_index()

    fig_map = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=latest_prices['geojson_name'],
        featureidkey="properties.Propinsi",
        z=latest_prices['price'],
        text=latest_prices['province'],
        colorscale="YlOrRd",
        colorbar_title="IDR/kg",
        hovertemplate="<b>%{text}</b><br>Harga: IDR %{z:,.0f}/kg<extra></extra>",
        marker_line_width=0.5,
        marker_line_color='rgba(255,255,255,0.3)',
    ))

    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor='rgba(0,0,0,0)',
    )
    apply_theme_to_plotly(
        fig_map,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Peta harga {selected_commodity} di Indonesia",
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Peta tidak bisa ditampilkan karena file batas provinsi belum ada. "
               "Letakkan file GeoJSON di `assets/indonesia.geojson`, lalu muat ulang halaman.")

# --- Price Disparity Analysis ---
st.subheader("Disparitas harga antarprovinsi")

latest_all = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
latest_all = latest_all.sort_values('price', ascending=False)

col1, col2, col3 = st.columns(3)

nat_avg = latest_all['price'].mean()
nat_std = latest_all['price'].std()
cv = (nat_std / nat_avg * 100) if nat_avg > 0 else 0

col1.metric("Rata-rata nasional", f"IDR {nat_avg:,.0f}/kg")
col2.metric("Simpangan baku", f"IDR {nat_std:,.0f}")
col3.metric("Koefisien variasi", f"{cv:.1f}%")
col3.caption("Disparitas " + ("tinggi (> 15%)" if cv > 15 else ("sedang (8-15%)" if cv > 8 else "rendah (< 8%)")))

# Disparity bar chart
col_a, col_b = st.columns([2, 1])

with col_a:
    colors = [STATUS['danger']['bg'] if p > nat_avg + nat_std else
              (STATUS['alert']['bg'] if p > nat_avg else STATUS['normal']['bg'])
              for p in latest_all['price']]

    fig_bar = go.Figure(go.Bar(
        x=latest_all['province'], y=latest_all['price'],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>IDR %{y:,.0f}/kg<extra></extra>",
    ))
    fig_bar.add_hline(y=nat_avg, line_dash="dash", line_color=CHART['reference'],
                      annotation_text=f"Rata-rata: IDR {nat_avg:,.0f}")
    apply_theme_to_plotly(
        fig_bar, height=400,
        title="Harga per provinsi: merah > rata-rata + 1σ, oranye > rata-rata, hijau ≤ rata-rata",
    )
    fig_bar.update_xaxes(showgrid=False, tickangle=45)
    fig_bar.update_yaxes(title='Harga (IDR/kg)')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.markdown("**5 provinsi termahal**")
    for _, row in latest_all.head(5).iterrows():
        diff = (row['price'] - nat_avg) / nat_avg * 100
        st.markdown(f"- **{row['province']}**: IDR {row['price']:,.0f} ({diff:+.1f}% dari rata-rata)")

    st.markdown("**5 provinsi termurah**")
    for _, row in latest_all.tail(5).iterrows():
        diff = (row['price'] - nat_avg) / nat_avg * 100
        st.markdown(f"- **{row['province']}**: IDR {row['price']:,.0f} ({diff:+.1f}% dari rata-rata)")

# --- Provincial Drill-down ---
st.divider()
st.subheader("Rincian per provinsi")

drill_province = st.selectbox("Provinsi", sorted(df['province'].unique()))

prov_series = df[(df['province'] == drill_province) & (df['commodity'] == selected_commodity)].sort_values('date')

if not prov_series.empty:
    c1, c2 = st.columns(2)
    
    with c1:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=prov_series['date'].tail(180), y=prov_series['price'].tail(180),
            mode='lines', name='Harga',
            line=dict(color=CHART['actual'], width=2),
        ))
        # Moving average
        ma30 = prov_series['price'].tail(180).rolling(30).mean()
        fig_trend.add_trace(go.Scatter(
            x=prov_series['date'].tail(180), y=ma30,
            mode='lines', name='Rata-rata bergerak 30 hari',
            line=dict(color=CHART['forecast'], width=2, dash='dash'),
        ))
        apply_theme_to_plotly(
            fig_trend, height=350,
            title=f"Harga 180 hari terakhir di {drill_province}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        # All commodities in this province
        prov_all = df[df['province'] == drill_province].groupby('commodity').last().reset_index()
        fig_comm = px.bar(
            prov_all.sort_values('price', ascending=True), x='price', y='commodity',
            orientation='h', color='price', color_continuous_scale='Viridis',
            labels={'price': 'Harga (IDR/kg)', 'commodity': ''},
            title=f"Harga terakhir semua komoditas di {drill_province}",
        )
        apply_theme_to_plotly(
            fig_comm, height=350, showlegend=False,
        )
        st.plotly_chart(fig_comm, use_container_width=True)
