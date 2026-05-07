"""
Page 1: Main Dashboard — Forecast, EWS, and Supply Risk Analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_store
from models.prophet_forecast import FoodPriceProphet
from models.lstm_forecast import LSTMForecaster
from engine.ews_engine_v2 import EWSEngineV2
from engine.supply_risk import SupplyRiskScorer

st.set_page_config(page_title="Dashboard | Agri-AI EWS", page_icon="🏠", layout="wide")

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data():
    store = get_store()
    df = store.load_all()
    if df.empty:
        csv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "food_prices_real.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df

df = load_data()
if df.empty:
    st.error("❌ Tidak ada data. Pastikan file food_prices_real.csv tersedia.")
    st.stop()

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2534/2534044.png", width=50)
st.sidebar.title("📊 Control Panel")
st.sidebar.markdown("---")

selected_province = st.sidebar.selectbox("🗺️ Provinsi", sorted(df['province'].unique()), index=min(10, len(df['province'].unique())-1))
selected_commodity = st.sidebar.selectbox("🌽 Komoditas", sorted(df['commodity'].unique()), index=0)

max_date = df['date'].max() + datetime.timedelta(days=120)
min_date = df['date'].max() + datetime.timedelta(days=1)
forecast_date = st.sidebar.date_input(
    "📅 Target Prediksi",
    value=min_date + datetime.timedelta(days=29),
    min_value=min_date,
    max_value=max_date
)

model_choice = st.sidebar.selectbox("🤖 Model AI", ["Hybrid (Prophet + BiLSTM)", "Prophet Only", "BiLSTM Only"])

st.sidebar.markdown("---")

# --- AI Forecast ---
@st.cache_resource(show_spinner=False)
def get_ai_forecast(_df, province, commodity, target_date, model_type):
    try:
        from models.evaluation import calculate_metrics
        from prophet import Prophet

        p_forecaster = FoodPriceProphet(_df)
        p_df = p_forecaster.prepare_data(province, commodity)
        train_df, test_df = p_forecaster.split_data(p_df, test_size=0.2)

        eval_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        eval_model.fit(train_df)
        eval_forecast = eval_model.predict(test_df[['ds']])
        metrics = calculate_metrics(test_df['y'].values, eval_forecast['yhat'].values, model_name="Prophet")

        p_forecast = p_forecaster.train_and_forecast(province, commodity, periods=120)

        target_dt = pd.to_datetime(target_date)
        p_row = p_forecast[p_forecast['ds'].dt.date == target_dt.date()]
        p_pred = p_row['yhat'].iloc[0] if not p_row.empty else p_forecast['yhat'].iloc[-1]
        p_lower = p_row['yhat_lower'].iloc[0] if not p_row.empty else p_forecast['yhat_lower'].iloc[-1]
        p_upper = p_row['yhat_upper'].iloc[0] if not p_row.empty else p_forecast['yhat_upper'].iloc[-1]

        if model_type == "Prophet Only":
            return float(p_pred), float(p_lower), float(p_upper), p_forecast, metrics

        # LSTM
        l_forecaster = LSTMForecaster(seq_length=30)
        X, y = l_forecaster.prepare_data(_df, province, commodity)
        l_forecaster.train_single_series(X[-300:], y[-300:], epochs=5)
        last_30 = _df[(_df['province'] == province) & (_df['commodity'] == commodity)]['price'].values[-30:]
        l_pred_next = l_forecaster.predict(last_30)[0][0]

        if model_type == "BiLSTM Only":
            return float(l_pred_next), float(l_pred_next * 0.92), float(l_pred_next * 1.08), p_forecast, metrics

        # Hybrid
        days_ahead = (target_dt.date() - _df['date'].max().date()).days
        weight_l = max(0.05, 0.4 - (days_ahead * 0.003))
        hybrid_pred = (p_pred * (1 - weight_l)) + (l_pred_next * weight_l)
        hybrid_lower = (p_lower * (1 - weight_l)) + (l_pred_next * 0.92 * weight_l)
        hybrid_upper = (p_upper * (1 - weight_l)) + (l_pred_next * 1.08 * weight_l)

        return float(hybrid_pred), float(hybrid_lower), float(hybrid_upper), p_forecast, metrics

    except Exception as e:
        import traceback
        st.error(f"⚠️ AI Engine Error: {e}\n{traceback.format_exc()}")
        return None, None, None, None, None

model_type_map = {
    "Hybrid (Prophet + BiLSTM)": "Hybrid",
    "Prophet Only": "Prophet Only",
    "BiLSTM Only": "BiLSTM Only",
}

with st.spinner(f"🧠 AI sedang menghitung prediksi untuk {forecast_date}..."):
    predicted_price, pred_lower, pred_upper, p_forecast, metrics = get_ai_forecast(
        df, selected_province, selected_commodity, forecast_date,
        model_type_map[model_choice]
    )

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏠 Dashboard Utama")
    st.markdown(f"**{forecast_date.strftime('%d %b %Y')}** | **{selected_commodity}** di **{selected_province}**")

# --- EWS v2 ---
current_data = df[(df['province'] == selected_province) & (df['commodity'] == selected_commodity)].sort_values('date')
current_price = current_data['price'].iloc[-1]

if predicted_price is not None:
    ews = EWSEngineV2(df)
    ews_result = ews.calculate_composite_score(selected_province, selected_commodity, predicted_price, forecast_date)
    supply_scorer = SupplyRiskScorer(df)
    supply_risk = supply_scorer.calculate_risk_score(selected_province, selected_commodity)
else:
    ews_result = {'level': 'Unknown', 'score': 0, 'message': 'AI Model offline', 'color': '#666', 'factors': {}, 'recommendations': []}
    supply_risk = {'score': 0, 'trend_direction': 'N/A', 'description': 'N/A', 'factors': {}}

with col2:
    level = ews_result.get('level', 'Unknown')
    score = ews_result.get('score', 0)
    
    # Level-specific styling
    level_config = {
        'Danger': {'gradient': 'linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%)', 'icon': '🔴', 'glow': 'rgba(255,65,108,0.4)'},
        'Alert':  {'gradient': 'linear-gradient(135deg, #F7971E 0%, #FFD200 100%)', 'icon': '🟠', 'glow': 'rgba(247,151,30,0.4)'},
        'Watch':  {'gradient': 'linear-gradient(135deg, #F2C94C 0%, #F2994A 100%)', 'icon': '🟡', 'glow': 'rgba(242,201,76,0.3)'},
        'Normal': {'gradient': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)', 'icon': '🟢', 'glow': 'rgba(56,239,125,0.3)'},
    }
    cfg = level_config.get(level, {'gradient': 'linear-gradient(135deg, #666 0%, #888 100%)', 'icon': '⚪', 'glow': 'rgba(100,100,100,0.3)'})
    
    st.markdown(f"""
        <div style="
            background: {cfg['gradient']};
            border-radius: 16px;
            padding: 22px 20px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px {cfg['glow']}, inset 0 1px 0 rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.15);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute; top: -20px; right: -20px;
                font-size: 5rem; opacity: 0.12;
                transform: rotate(15deg);
            ">⚠️</div>
            <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85; font-weight: 600;">
                EWS Status
            </div>
            <div style="font-size: 1.8rem; font-weight: 900; margin: 4px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                {cfg['icon']} {level.upper()}
            </div>
            <div style="
                font-size: 2rem; font-weight: 900;
                background: rgba(0,0,0,0.15);
                border-radius: 10px;
                padding: 4px 0;
                margin: 6px 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">{score}/100</div>
            <div style="
                font-size: 0.72rem;
                opacity: 0.9;
                margin-top: 6px;
                line-height: 1.3;
                padding: 0 5px;
            ">{ews_result.get('message', '')[:80]}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Dynamic Metrics ---
m1, m2, m3, m4 = st.columns(4)

# Price change (7 days)
if len(current_data) >= 7:
    price_7d = current_data['price'].iloc[-7]
    pct_7d = (current_price - price_7d) / price_7d * 100
else:
    pct_7d = 0
m1.metric("Harga Pasar Terakhir", f"IDR {current_price:,.0f}/kg", f"{pct_7d:+.1f}% (7d)")

if predicted_price is not None:
    price_diff = (predicted_price - current_price) / current_price * 100
    m2.metric(f"Prediksi ({forecast_date.strftime('%d %b')})", f"IDR {predicted_price:,.0f}/kg", f"{price_diff:+.1f}%", delta_color="inverse")
else:
    m2.metric(f"Prediksi ({forecast_date.strftime('%d %b')})", "N/A", "0.0%")

volatility = current_data['price'].pct_change().std() * 100
vol_7d_ago = current_data['price'].iloc[:-7].pct_change().std() * 100 if len(current_data) > 14 else volatility
vol_change = volatility - vol_7d_ago
m3.metric("Volatilitas Pasar", f"{volatility:.2f}%", f"{vol_change:+.2f}%")

m4.metric("Supply Risk Score", f"{supply_risk['score']:.0f}/100", supply_risk['trend_direction'])

# --- Charts ---
st.markdown("### 📊 Market Intelligence")
tab1, tab2, tab3, tab4 = st.tabs(["📉 Forecast", "📍 Regional", "🔍 Correlation", "🔬 Model"])

with tab1:
    if p_forecast is not None:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=current_data['date'].tail(90), y=current_data['price'].tail(90),
            mode='lines+markers', name='Historical (90d)',
            line=dict(color='#4facfe', width=3), marker=dict(size=3)
        ))

        future_data = p_forecast[p_forecast['ds'] > current_data['date'].max()]
        
        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([future_data['ds'], future_data['ds'][::-1]]),
            y=pd.concat([future_data['yhat_upper'], future_data['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255,165,0,0.1)',
            line=dict(color='rgba(255,165,0,0)'),
            name='90% Confidence Interval'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_data['ds'], y=future_data['yhat'],
            mode='lines', name='Prophet Forecast',
            line=dict(color='#FFA500', width=2, dash='dot')
        ))

        if predicted_price is not None:
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(forecast_date)], y=[predicted_price],
                mode='markers', name=f'{model_choice} Target',
                marker=dict(color='#FF4B4B', size=14, symbol='star',
                           line=dict(width=2, color='white'))
            ))

            # Confidence range for target
            if pred_lower and pred_upper:
                fig.add_trace(go.Scatter(
                    x=[pd.Timestamp(forecast_date)]*2,
                    y=[pred_lower, pred_upper],
                    mode='lines', name='Prediction Range',
                    line=dict(color='#FF4B4B', width=3),
                ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Harga (IDR/kg)'),
            height=500, margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Chart tidak tersedia. AI model belum terhubung.")

with tab2:
    latest_all = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
    fig_comp = px.bar(
        latest_all.sort_values('price', ascending=False), x='province', y='price',
        color='price', title=f"Distribusi Harga: {selected_commodity}",
        template="plotly_dark", color_continuous_scale="Viridis",
        labels={'price': 'Harga (IDR/kg)', 'province': ''}
    )
    fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Korelasi Antar-Komoditas**")
        prov_data = df[df['province'] == selected_province].pivot(index='date', columns='commodity', values='price')
        corr = prov_data.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                             title=f"Matriks Korelasi — {selected_province}")
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)
    with col_b:
        st.write("**Analisis Faktor EWS**")
        factors = ews_result.get('factors', {})
        factor_names = {
            'price_change': '📈 Perubahan Harga',
            'volatility': '📊 Volatilitas',
            'seasonal': '📅 Anomali Musiman',
            'cross_region': '🗺️ Anomali Regional',
            'velocity': '🚀 Kecepatan Perubahan',
        }
        for key, score in factors.items():
            name = factor_names.get(key, key)
            color_bar = '#FF4B4B' if score > 60 else ('#FFA500' if score > 30 else '#00CC96')
            st.markdown(f"**{name}**: {score:.0f}/100")
            st.progress(min(score / 100, 1.0))

        st.markdown("---")
        st.write("**Rekomendasi:**")
        for rec in ews_result.get('recommendations', []):
            st.markdown(f"- {rec}")

with tab4:
    st.markdown("### Evaluasi Model AI")
    if metrics is not None:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("📉 RMSE", f"{metrics['RMSE']:,.2f}")
        mc2.metric("📉 MAE", f"{metrics['MAE']:,.2f}")
        mc3.metric("🎯 MAPE", f"{metrics['MAPE (%)']:.2f}%")

        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("📐 R²", f"{metrics.get('R²', 0):.4f}")
        mc5.metric("📊 SMAPE", f"{metrics.get('SMAPE (%)', 0):.2f}%")
        mc6.metric("🎯 Directional Acc.", f"{metrics.get('Directional Accuracy (%)', 0):.1f}%")

        mape = metrics['MAPE (%)']
        if mape < 5:
            grade = "🏆 Exceptional (< 5%)"
        elif mape < 10:
            grade = "✅ Highly Accurate (< 10%)"
        elif mape < 20:
            grade = "👍 Good (< 20%)"
        else:
            grade = "⚠️ Needs Improvement (> 20%)"
        st.success(f"**Grade: {grade}** — MAPE {mape:.2f}%")
    else:
        st.info("⚠️ Metrik belum tersedia.")

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; opacity: 0.5; font-size: 0.8rem;">
    <div>ENGINE: PROPHET + BiLSTM + TFT</div>
    <div>DATA: PIHPS Bank Indonesia</div>
    <div>© 2026 Fahmi Prasanda</div>
</div>
""", unsafe_allow_html=True)
