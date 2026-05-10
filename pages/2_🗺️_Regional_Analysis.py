"""
Page 2: Regional Analysis — Indonesia Choropleth Map & Provincial Drill-down.
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

st.set_page_config(page_title="Regional Analysis | Agri-AI EWS", page_icon="🗺️", layout="wide")

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
    st.error("❌ Data tidak tersedia.")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🗺️ Regional Analysis")
st.sidebar.markdown("---")
selected_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), index=0)
st.sidebar.markdown("---")

# --- Main Content ---
st.title("🗺️ Regional Analysis")
st.markdown(f"### Analisis Harga **{selected_commodity}** Seluruh Indonesia")

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
    fig_map.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Peta Harga {selected_commodity} — Indonesia",
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("⚠️ GeoJSON file not found. Install it at `assets/indonesia.geojson`.")

# --- Price Disparity Analysis ---
st.markdown("### 📊 Analisis Disparitas Harga")

latest_all = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
latest_all = latest_all.sort_values('price', ascending=False)

col1, col2, col3 = st.columns(3)

nat_avg = latest_all['price'].mean()
nat_std = latest_all['price'].std()
cv = (nat_std / nat_avg * 100) if nat_avg > 0 else 0

col1.metric("Rata-rata Nasional", f"IDR {nat_avg:,.0f}/kg")
col2.metric("Std Deviasi", f"IDR {nat_std:,.0f}")
col3.metric("Coefficient of Variation", f"{cv:.1f}%", 
            "Tinggi" if cv > 15 else ("Sedang" if cv > 8 else "Rendah"))

# Disparity bar chart
col_a, col_b = st.columns([2, 1])

with col_a:
    colors = ['#FF4B4B' if p > nat_avg + nat_std else 
              ('#FFA500' if p > nat_avg else '#00CC96') 
              for p in latest_all['price']]
    
    fig_bar = go.Figure(go.Bar(
        x=latest_all['province'], y=latest_all['price'],
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>IDR %{y:,.0f}/kg<extra></extra>",
    ))
    fig_bar.add_hline(y=nat_avg, line_dash="dash", line_color="white", opacity=0.5,
                      annotation_text=f"Rata-rata: IDR {nat_avg:,.0f}")
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', height=400,
        xaxis=dict(showgrid=False, tickangle=45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Harga (IDR/kg)'),
        title="Harga per Provinsi (merah = di atas rata-rata + 1σ)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.markdown("**🔴 Provinsi Termahal**")
    for _, row in latest_all.head(5).iterrows():
        diff = (row['price'] - nat_avg) / nat_avg * 100
        st.markdown(f"- **{row['province']}**: IDR {row['price']:,.0f} (+{diff:.1f}%)")

    st.markdown("**🟢 Provinsi Termurah**")
    for _, row in latest_all.tail(5).iterrows():
        diff = (row['price'] - nat_avg) / nat_avg * 100
        st.markdown(f"- **{row['province']}**: IDR {row['price']:,.0f} ({diff:.1f}%)")

# --- Provincial Drill-down ---
st.markdown("---")
st.markdown("### 🔍 Drill-Down Provinsi")

drill_province = st.selectbox("Pilih Provinsi untuk Analisis Detail", sorted(df['province'].unique()))

prov_series = df[(df['province'] == drill_province) & (df['commodity'] == selected_commodity)].sort_values('date')

if not prov_series.empty:
    c1, c2 = st.columns(2)
    
    with c1:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=prov_series['date'].tail(180), y=prov_series['price'].tail(180),
            mode='lines', name='Harga',
            line=dict(color='#4facfe', width=2),
            fill='tozeroy', fillcolor='rgba(79,172,254,0.1)',
        ))
        # Moving average
        ma30 = prov_series['price'].tail(180).rolling(30).mean()
        fig_trend.add_trace(go.Scatter(
            x=prov_series['date'].tail(180), y=ma30,
            mode='lines', name='MA-30',
            line=dict(color='#FFA500', width=2, dash='dash'),
        ))
        fig_trend.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=350,
            title=f"Tren Harga 180 Hari — {drill_province}",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        # All commodities in this province
        prov_all = df[df['province'] == drill_province].groupby('commodity').last().reset_index()
        fig_comm = px.bar(
            prov_all.sort_values('price', ascending=True), x='price', y='commodity',
            orientation='h', color='price', color_continuous_scale='Viridis',
            labels={'price': 'Harga (IDR/kg)', 'commodity': ''},
            title=f"Semua Komoditas — {drill_province}",
        )
        fig_comm.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=350, showlegend=False,
        )
        st.plotly_chart(fig_comm, use_container_width=True)
