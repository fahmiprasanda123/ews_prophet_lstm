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
from models.tft_forecast import get_tft_forecaster
from models.ensemble import SmartEnsemble
from engine.ews_engine_v2 import EWSEngineV2
from engine.supply_risk import SupplyRiskScorer
from engine.price_narrative import PriceNarrativeAnalyzer
from models.evaluation import calculate_metrics
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

st.set_page_config(page_title="Dashboard | Agri-AI EWS", page_icon="🏠", layout="wide")

# --- Theme ---
from theme import inject_theme_css, render_theme_toggle, theme_color, get_plotly_template, get_plotly_layout, get_plotly_yaxis, apply_theme_to_plotly
inject_theme_css()

# --- Initialize Session State if not present ---
if 'model_params' not in st.session_state:
    st.session_state.model_params = {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'epochs': 50,
        'hidden_size': 128,
        'seq_length': 30,
        'tft_max_epochs': 15,
        'tft_batch_size': 64
    }

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
render_theme_toggle()
st.sidebar.markdown("---")

selected_province = st.sidebar.selectbox("🗺️ Provinsi", sorted(df['province'].unique()), index=min(10, len(df['province'].unique())-1))
selected_commodity = st.sidebar.selectbox("🌽 Komoditas", sorted(df['commodity'].unique()), index=0)

# Horizon dibatasi 30 hari dari data terakhir, sesuai horizon yang divalidasi
# protokol evaluasi. Prakiraan rekursif BiLSTM melenceng jauh di luar horizon ini.
MAX_HORIZON_DAYS = 30
last_data_date = df['date'].max().date()
today = datetime.date.today()
min_date = last_data_date + datetime.timedelta(days=1)
max_date_val = last_data_date + datetime.timedelta(days=MAX_HORIZON_DAYS)
forecast_date = st.sidebar.date_input(
    "📅 Target Prediksi",
    value=max_date_val,
    min_value=min_date,
    max_value=max_date_val,
    help=f"Maksimal {MAX_HORIZON_DAYS} hari setelah data terakhir ({last_data_date:%d %b %Y})."
)
if (today - last_data_date).days > 7:
    st.sidebar.warning(f"Data terakhir {last_data_date:%d %b %Y}, tertinggal {(today - last_data_date).days} hari. "
                       "Sinkronkan data agar prakiraan mencakup tanggal sekarang.")

model_choice = st.sidebar.selectbox("🤖 Model AI", ["Smart Ensemble (All Models)", "Hybrid (Prophet + BiLSTM)", "TFT (Transformer)", "Prophet Only", "BiLSTM Only"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Parameters")

with st.sidebar.expander("🔮 Prophet Config", expanded=False):
    p_cps = st.slider("Changepoint Prior Scale", 0.001, 0.5, st.session_state.model_params['changepoint_prior_scale'], format="%.3f")
    p_yearly = st.checkbox("Yearly Seasonality", st.session_state.model_params['yearly_seasonality'])
    p_weekly = st.checkbox("Weekly Seasonality", st.session_state.model_params['weekly_seasonality'])

with st.sidebar.expander("🧠 BiLSTM Config", expanded=False):
    l_epochs = st.number_input("Epoch maksimum (early stopping)", 5, 200, st.session_state.model_params['epochs'])
    l_hidden = st.selectbox("Hidden Size", [32, 64, 128, 256], index=[32, 64, 128, 256].index(st.session_state.model_params['hidden_size']))
    l_seq = st.slider("Sequence Length", 7, 60, st.session_state.model_params['seq_length'])

with st.sidebar.expander("⚡ TFT Config", expanded=False):
    t_epochs = st.number_input("Max Epochs", 1, 50, st.session_state.model_params['tft_max_epochs'])
    t_batch = st.selectbox("Batch Size", [16, 32, 64], index=[16, 32, 64].index(st.session_state.model_params['tft_batch_size']))

# Update session state
st.session_state.model_params = {
    'changepoint_prior_scale': p_cps,
    'yearly_seasonality': p_yearly,
    'weekly_seasonality': p_weekly,
    'epochs': l_epochs,
    'hidden_size': l_hidden,
    'seq_length': l_seq,
    'tft_max_epochs': t_epochs,
    'tft_batch_size': t_batch
}


st.sidebar.markdown("---")

# --- AI Forecast ---
MODELS_FOR = {
    "Prophet Only": ['prophet'],
    "BiLSTM Only": ['bilstm'],
    "TFT": ['tft'],
    "Hybrid": ['prophet', 'bilstm', 'ensemble'],
    "Smart Ensemble": ['prophet', 'bilstm', 'tft', 'ensemble'],
}
EVAL_NAME = {"Prophet Only": "Prophet", "BiLSTM Only": "BiLSTM", "TFT": "TFT",
             "Hybrid": "Smart Ensemble", "Smart Ensemble": "Smart Ensemble"}
WEIGHT_KEY = {'Prophet': 'prophet', 'BiLSTM': 'lstm', 'TFT': 'tft'}


def _protocol_config(params):
    return {
        'prophet': {'changepoint_prior_scale': float(params['changepoint_prior_scale']),
                    'yearly_seasonality': bool(params['yearly_seasonality']),
                    'weekly_seasonality': bool(params['weekly_seasonality'])},
        'lstm': {'epochs': int(params['epochs']), 'hidden_size': int(params['hidden_size']),
                 'seq_length': int(params['seq_length'])},
        'tft': {'max_epochs': int(params['tft_max_epochs']), 'batch_size': int(params['tft_batch_size'])},
    }


@st.cache_resource(show_spinner=False)
def get_model_evaluation(_df, province, commodity, model_type, params):
    """Metrik dan bobot ensemble dari protokol evaluasi yang sama dengan Model Laboratory/skrip tesis."""
    from models.evaluation_protocol import evaluate_series
    return evaluate_series(_df, province, commodity, config=_protocol_config(params),
                           models=MODELS_FOR[model_type])


@st.cache_resource(show_spinner=False)
def get_ai_forecast(_df, province, commodity, target_date, model_type, params):
    try:
        from models.ensemble import SmartEnsemble
        from models.lstm_forecast import LSTMForecaster

        series = _df[(_df['province'] == province) & (_df['commodity'] == commodity)].sort_values('date')
        target_dt = pd.to_datetime(target_date)
        last_date = pd.to_datetime(series['date'].max())
        days_ahead = max(1, (target_dt.normalize() - last_date.normalize()).days)
        notes = []

        # 1. Prophet (selalu, untuk grafik tren jangka panjang)
        p_forecaster = FoodPriceProphet(_df)
        p_forecast = p_forecaster.train_and_forecast(province, commodity, periods=max(120, days_ahead))
        p_row = p_forecast[p_forecast['ds'].dt.date == target_dt.date()]
        p_row = p_row if not p_row.empty else p_forecast.tail(1)
        future_preds = {}
        if 'prophet' in MODELS_FOR[model_type]:
            future_preds['prophet'] = {'mean': float(p_row['yhat'].iloc[0]),
                                       'lower': float(p_row['yhat_lower'].iloc[0]),
                                       'upper': float(p_row['yhat_upper'].iloc[0])}

        # 2. BiLSTM: prakiraan rekursif sampai tanggal target + interval MC Dropout
        if 'bilstm' in MODELS_FOR[model_type]:
            lf = LSTMForecaster(seq_length=params['seq_length'], hidden_size=params['hidden_size'])
            X_all, y_all = lf.prepare_data(_df, province, commodity)
            lf.train_single_series(X_all, y_all, epochs=params['epochs'])
            last_seq = series['price'].values[-params['seq_length']:]
            point = lf.predict_multi_step(last_seq, steps=days_ahead)
            mc = lf.predict_with_uncertainty(last_seq, steps=days_ahead, n_samples=50)
            future_preds['lstm'] = {'mean': float(point[-1]), 'lower': float(mc['lower'][-1]),
                                    'upper': float(mc['upper'][-1])}

        # 3. TFT: prakiraan ke DEPAN (bukan 30 hari terakhir data)
        if 'tft' in MODELS_FOR[model_type]:
            tft_model = get_tft_forecaster()
            if tft_model.is_available:
                dataset, data = tft_model.prepare_dataset(_df, province, commodity, use_all_data=True)
                if dataset is not None:
                    tft_model.train(dataset, max_epochs=params['tft_max_epochs'],
                                    batch_size=params['tft_batch_size'], quiet=True)
                    fut = tft_model.forecast_future(data, dataset)
                    idx = min(days_ahead, len(fut['mean'])) - 1
                    if days_ahead > len(fut['mean']):
                        notes.append(f"TFT hanya memprakirakan {len(fut['mean'])} hari; dipakai nilai hari terakhirnya.")
                    future_preds['tft'] = {'mean': float(fut['mean'][idx]), 'lower': float(fut['lower'][idx]),
                                           'upper': float(fut['upper'][idx])}
            else:
                notes.append("TFT tidak tersedia (pytorch-forecasting belum terpasang).")

        # 4. Metrik dan bobot dari protokol evaluasi (di-cache per kombinasi)
        evaluation = get_model_evaluation(_df, province, commodity, model_type, params)
        mrow = evaluation['metrics'][evaluation['metrics']['Model'] == EVAL_NAME[model_type]]
        metrics = mrow.iloc[0].to_dict() if len(mrow) else None

        ensemble_info = None
        if model_type in ("Smart Ensemble", "Hybrid") and evaluation.get('ensemble'):
            weights = {WEIGHT_KEY[k]: v for k, v in evaluation['ensemble']['weights'].items()}
            ens = SmartEnsemble(default_weights=weights)
            res = ens.combine_forecasts({k: v for k, v in future_preds.items() if k in weights})
            predicted_price = float(np.atleast_1d(res['mean'])[0])
            pred_lower = float(np.atleast_1d(res['lower'])[0])
            pred_upper = float(np.atleast_1d(res['upper'])[0])
            ensemble_info = {'weights': res['model_weights'], 'models_used': res['models_used'], 'notes': notes}
        else:
            key = {'Prophet Only': 'prophet', 'BiLSTM Only': 'lstm', 'TFT': 'tft'}.get(model_type)
            if key not in future_preds:
                return None, None, None, p_forecast, metrics, None
            single = future_preds[key]
            predicted_price, pred_lower, pred_upper = single['mean'], single['lower'], single['upper']

        return float(predicted_price), float(pred_lower), float(pred_upper), p_forecast, metrics, ensemble_info

    except Exception as e:
        import traceback
        st.error(f"⚠️ AI Engine Error: {e}\n{traceback.format_exc()}")
        return None, None, None, None, None, None

model_type_map = {
    "Smart Ensemble (All Models)": "Smart Ensemble",
    "Hybrid (Prophet + BiLSTM)": "Hybrid",
    "TFT (Transformer)": "TFT",
    "Prophet Only": "Prophet Only",
    "BiLSTM Only": "BiLSTM Only",
}

with st.spinner(f"🧠 AI sedang menghitung prediksi untuk {forecast_date}..."):
    predicted_price, pred_lower, pred_upper, p_forecast, metrics, ensemble_info = get_ai_forecast(
        df, selected_province, selected_commodity, forecast_date,
        model_type_map[model_choice], st.session_state.model_params
    )

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏠 Dashboard Utama")
    st.markdown(f"**{forecast_date.strftime('%d %b %Y')}** | **{selected_commodity}** di **{selected_province}**")
    
    if ensemble_info:
        st.markdown("### 🎯 Smart Ensemble Active")
        cols = st.columns(len(ensemble_info['weights']))
        for i, (model_name, weight) in enumerate(ensemble_info['weights'].items()):
            cols[i].metric(model_name.upper(), f"{weight*100:.1f}%")


# --- EWS v2 ---
current_data = df[(df['province'] == selected_province) & (df['commodity'] == selected_commodity)].sort_values('date')
current_price = current_data['price'].iloc[-1]

if predicted_price is not None:
    ews = EWSEngineV2(df)
    ews_result = ews.calculate_composite_score(selected_province, selected_commodity, predicted_price, forecast_date)
    supply_scorer = SupplyRiskScorer(df)
    supply_risk = supply_scorer.calculate_risk_score(selected_province, selected_commodity)
    # Generate narrative analysis
    narrator = PriceNarrativeAnalyzer(df)
    narrative_result = narrator.generate_narrative(selected_province, selected_commodity, predicted_price, forecast_date)
else:
    ews_result = {'level': 'Unknown', 'score': 0, 'message': 'AI Model offline', 'color': '#666', 'factors': {}, 'recommendations': []}
    supply_risk = {'score': 0, 'trend_direction': 'N/A', 'description': 'N/A', 'factors': {}}
    narrative_result = None

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

# --- Narrative Analysis Section ---
if narrative_result and narrative_result.get('direction') != 'UNKNOWN':
    st.markdown("### 📝 Analisis Penyebab Prediksi Harga")
    
    # Direction badge
    direction = narrative_result['direction']
    pct = narrative_result.get('pct_change', 0)
    dir_config = {
        'NAIK': {'icon': '🔺', 'color': '#FF4B4B', 'bg': 'rgba(255,75,75,0.1)', 'border': 'rgba(255,75,75,0.3)', 'label': 'HARGA DIPREDIKSI NAIK'},
        'TURUN': {'icon': '🔻', 'color': '#00CC96', 'bg': 'rgba(0,204,150,0.1)', 'border': 'rgba(0,204,150,0.3)', 'label': 'HARGA DIPREDIKSI TURUN'},
        'STABIL': {'icon': '➡️', 'color': '#4facfe', 'bg': 'rgba(79,172,254,0.1)', 'border': 'rgba(79,172,254,0.3)', 'label': 'HARGA DIPREDIKSI STABIL'},
    }
    dcfg = dir_config.get(direction, dir_config['STABIL'])
    
    st.markdown(f"""
        <div style="
            background: {dcfg['bg']};
            border: 2px solid {dcfg['border']};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="font-size: 2.5rem;">{dcfg['icon']}</div>
            <div>
                <div style="font-size: 1.1rem; font-weight: 800; color: {dcfg['color']}; letter-spacing: 1px;">
                    {dcfg['label']} ({pct:+.1f}%)
                </div>
                <div style="font-size: 0.9rem; opacity: 0.85; margin-top: 4px; line-height: 1.4;">
                    {narrative_result.get('summary', '')}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Factor details in expandable sections
    factors = narrative_result.get('factors', [])
    if factors:
        impact_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        impact_labels = {'high': 'Dampak Tinggi', 'medium': 'Dampak Sedang', 'low': 'Dampak Rendah'}
        
        # Show factors in columns for high-impact, then expanders for the rest
        high_factors = [f for f in factors if f['impact'] == 'high']
        other_factors = [f for f in factors if f['impact'] != 'high']
        
        if high_factors:
            for f in high_factors:
                impact_icon = impact_icons.get(f['impact'], '⚪')
                st.markdown(f"""
                    <div style="
                        background: rgba(255,75,75,0.08);
                        border-left: 4px solid #FF4B4B;
                        border-radius: 0 8px 8px 0;
                        padding: 14px 18px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-weight: 700; margin-bottom: 6px;">
                            {impact_icon} {f['name']} — <span style="color: #FF4B4B; font-size: 0.8rem;">{impact_labels.get(f['impact'], '')}</span>
                        </div>
                        <div style="font-size: 0.85rem; line-height: 1.6; opacity: 0.9;">
                            {f['description']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        if other_factors:
            with st.expander(f"📋 Lihat {len(other_factors)} faktor lainnya", expanded=False):
                for f in other_factors:
                    impact_icon = impact_icons.get(f['impact'], '⚪')
                    border_color = '#FFD700' if f['impact'] == 'medium' else '#00CC96'
                    bg_color = 'rgba(255,215,0,0.06)' if f['impact'] == 'medium' else 'rgba(0,204,150,0.06)'
                    st.markdown(f"""
                        <div style="
                            background: {bg_color};
                            border-left: 4px solid {border_color};
                            border-radius: 0 8px 8px 0;
                            padding: 12px 16px;
                            margin-bottom: 8px;
                        ">
                            <div style="font-weight: 700; margin-bottom: 4px;">
                                {impact_icon} {f['name']} — <span style="font-size: 0.8rem; color: {border_color};">{impact_labels.get(f['impact'], '')}</span>
                            </div>
                            <div style="font-size: 0.85rem; line-height: 1.6; opacity: 0.9;">
                                {f['description']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    # Full narrative in expander
    with st.expander("📖 Baca Narasi Analisis Lengkap", expanded=False):
        st.markdown(narrative_result.get('narrative', ''))

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

        apply_theme_to_plotly(
            fig,
            height=500, margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(title='Harga (IDR/kg)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Chart tidak tersedia. AI model belum terhubung.")

with tab2:
    latest_all = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
    fig_comp = px.bar(
        latest_all.sort_values('price', ascending=False), x='province', y='price',
        color='price', title=f"Distribusi Harga: {selected_commodity}",
        color_continuous_scale="Viridis",
        labels={'price': 'Harga (IDR/kg)', 'province': ''}
    )
    apply_theme_to_plotly(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Korelasi Antar-Komoditas**")
        prov_data = df[df['province'] == selected_province].pivot(index='date', columns='commodity', values='price')
        corr = prov_data.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                             title=f"Matriks Korelasi — {selected_province}")
        apply_theme_to_plotly(fig_corr)
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
        st.success(f"**Kategori (Lewis, 1982): {metrics.get('Kategori MAPE', '-')}** — MAPE {mape:.2f}%")
        st.caption("Metrik dihitung dengan protokol rolling-origin (split 80/20, horizon 30 hari) yang sama "
                   "dengan Model Laboratory. Directional Accuracy relatif terhadap harga di titik asal.")
    else:
        st.info("⚠️ Metrik belum tersedia.")

# Footer
st.markdown("---")
st.markdown("""
<div class="theme-footer">
    <div>ENGINE: PROPHET + BiLSTM + TFT</div>
    <div>DATA: PIHPS Bank Indonesia</div>
    <div>© 2026 Fahmi Prasanda</div>
</div>
""", unsafe_allow_html=True)
