"""
Page 1: Main Dashboard: Forecast, EWS, and Supply Risk Analysis.
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

st.set_page_config(page_title="Dashboard | Agri-AI EWS", page_icon="🌾", layout="wide")

# --- Theme ---
from theme import (inject_theme_css, render_theme_toggle, render_sidebar_brand, render_footer,
                   apply_theme_to_plotly, status_chip, ews_card, CHART)
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
    st.error("Belum ada data harga. Buka halaman utama untuk menjalankan sinkronisasi PIHPS, "
             "atau letakkan food_prices_real.csv di folder proyek lalu muat ulang halaman.")
    st.stop()

# --- Sidebar ---
render_sidebar_brand("Panel kontrol")
render_theme_toggle()
st.sidebar.divider()

selected_province = st.sidebar.selectbox("Provinsi", sorted(df['province'].unique()), index=min(10, len(df['province'].unique())-1))
selected_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), index=0)

# Horizon dibatasi 30 hari dari data terakhir, sesuai horizon yang divalidasi
# protokol evaluasi. Prakiraan rekursif BiLSTM melenceng jauh di luar horizon ini.
MAX_HORIZON_DAYS = 30
last_data_date = df['date'].max().date()
today = datetime.date.today()
min_date = last_data_date + datetime.timedelta(days=1)
max_date_val = last_data_date + datetime.timedelta(days=MAX_HORIZON_DAYS)
forecast_date = st.sidebar.date_input(
    "Tanggal target prediksi",
    value=max_date_val,
    min_value=min_date,
    max_value=max_date_val,
    help=f"Maksimal {MAX_HORIZON_DAYS} hari setelah data terakhir ({last_data_date:%d %b %Y})."
)
if (today - last_data_date).days > 7:
    st.sidebar.warning(f"Data terakhir {last_data_date:%d %b %Y}, tertinggal {(today - last_data_date).days} hari. "
                       "Sinkronkan data agar prakiraan mencakup tanggal sekarang.")

model_choice = st.sidebar.selectbox("Model", ["Ensemble (Prophet + BiLSTM + TFT)", "Hybrid (Prophet + BiLSTM)", "TFT", "Prophet saja", "BiLSTM saja"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Parameter model")

with st.sidebar.expander("Prophet", expanded=False):
    p_cps = st.slider("Changepoint prior scale", 0.001, 0.5, st.session_state.model_params['changepoint_prior_scale'], format="%.3f")
    p_yearly = st.checkbox("Musiman tahunan", st.session_state.model_params['yearly_seasonality'])
    p_weekly = st.checkbox("Musiman mingguan", st.session_state.model_params['weekly_seasonality'])

with st.sidebar.expander("BiLSTM", expanded=False):
    l_epochs = st.number_input("Epoch maksimum (early stopping)", 5, 200, st.session_state.model_params['epochs'])
    l_hidden = st.selectbox("Hidden size", [32, 64, 128, 256], index=[32, 64, 128, 256].index(st.session_state.model_params['hidden_size']))
    l_seq = st.slider("Panjang sekuens (hari)", 7, 60, st.session_state.model_params['seq_length'])

with st.sidebar.expander("TFT", expanded=False):
    t_epochs = st.number_input("Epoch maksimum", 1, 50, st.session_state.model_params['tft_max_epochs'])
    t_batch = st.selectbox("Batch size", [16, 32, 64], index=[16, 32, 64].index(st.session_state.model_params['tft_batch_size']))

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
        traceback.print_exc()
        st.error(f"Prakiraan gagal dihitung untuk {commodity} di {province}. "
                 "Coba model lain di sidebar atau kecilkan parameter model, lalu muat ulang halaman. "
                 f"(Detail teknis: {type(e).__name__}: {e})")
        return None, None, None, None, None, None

model_type_map = {
    "Ensemble (Prophet + BiLSTM + TFT)": "Smart Ensemble",
    "Hybrid (Prophet + BiLSTM)": "Hybrid",
    "TFT": "TFT",
    "Prophet saja": "Prophet Only",
    "BiLSTM saja": "BiLSTM Only",
}

with st.spinner(f"Menghitung prakiraan {selected_commodity} di {selected_province} untuk {forecast_date:%d %b %Y}. "
                "Pelatihan model bisa memakan beberapa menit."):
    predicted_price, pred_lower, pred_upper, p_forecast, metrics, ensemble_info = get_ai_forecast(
        df, selected_province, selected_commodity, forecast_date,
        model_type_map[model_choice], st.session_state.model_params
    )

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Dashboard")
    st.markdown(f"**{selected_commodity}** di **{selected_province}**, target **{forecast_date.strftime('%d %b %Y')}**")

    if ensemble_info:
        st.markdown("##### Bobot ensemble")
        cols = st.columns(len(ensemble_info['weights']))
        for i, (model_name, weight) in enumerate(ensemble_info['weights'].items()):
            cols[i].metric({'prophet': 'Prophet', 'lstm': 'BiLSTM', 'tft': 'TFT'}.get(model_name, model_name),
                           f"{weight*100:.1f}%")


# --- EWS v2 ---
current_data = df[(df['province'] == selected_province) & (df['commodity'] == selected_commodity)].sort_values('date')
current_price = current_data['price'].iloc[-1]

# Risiko pasokan dihitung dari data historis, tidak bergantung pada prakiraan
supply_risk = SupplyRiskScorer(df).calculate_risk_score(selected_province, selected_commodity)

if predicted_price is not None:
    ews = EWSEngineV2(df)
    ews_result = ews.calculate_composite_score(selected_province, selected_commodity, predicted_price, forecast_date)
    # Generate narrative analysis
    narrator = PriceNarrativeAnalyzer(df)
    narrative_result = narrator.generate_narrative(selected_province, selected_commodity, predicted_price, forecast_date)
else:
    ews_result = {'level': 'Unknown', 'score': 0,
                  'message': 'Status belum bisa dihitung karena prakiraan gagal. Lihat pesan di atas.',
                  'factors': {}, 'recommendations': []}
    narrative_result = None

with col2:
    # Kartu status: fokus utama layar, satu warna solid per level (DESIGN.md)
    st.markdown(ews_card(ews_result.get('level', 'Unknown'), ews_result.get('score', 0),
                         ews_result.get('message', '')), unsafe_allow_html=True)

# --- Dynamic Metrics ---
m1, m2, m3, m4 = st.columns(4)

# Price change (7 days)
if len(current_data) >= 7:
    price_7d = current_data['price'].iloc[-7]
    pct_7d = (current_price - price_7d) / price_7d * 100
else:
    pct_7d = 0
m1.metric("Harga pasar terakhir", f"IDR {current_price:,.0f}/kg", f"{pct_7d:+.1f}% dalam 7 hari", delta_color="inverse")

if predicted_price is not None:
    price_diff = (predicted_price - current_price) / current_price * 100
    m2.metric(f"Prediksi ({forecast_date.strftime('%d %b')})", f"IDR {predicted_price:,.0f}/kg", f"{price_diff:+.1f}% dari harga terakhir", delta_color="inverse")
else:
    m2.metric(f"Prediksi ({forecast_date.strftime('%d %b')})", "Belum tersedia")

volatility = current_data['price'].pct_change().std() * 100
vol_7d_ago = current_data['price'].iloc[:-7].pct_change().std() * 100 if len(current_data) > 14 else volatility
vol_change = volatility - vol_7d_ago
m3.metric("Volatilitas harian", f"{volatility:.2f}%", f"{vol_change:+.2f} poin vs 7 hari lalu", delta_color="inverse")

m4.metric("Skor risiko pasokan", f"{supply_risk['score']:.0f}/100")
if supply_risk.get('trend_direction'):
    m4.caption(f"Tren harga 7 hari: {supply_risk['trend_direction'].lower()}")

# --- Narrative Analysis Section ---
DIRECTION_STATUS = {'NAIK': ('Harga diprediksi naik', 'danger'),
                    'TURUN': ('Harga diprediksi turun', 'normal'),
                    'STABIL': ('Harga diprediksi stabil', 'neutral')}
IMPACT_STATUS = {'high': ('Dampak tinggi', 'danger'), 'medium': ('Dampak sedang', 'watch'),
                 'low': ('Dampak rendah', 'neutral')}


def _factor_html(f):
    label, status = IMPACT_STATUS.get(f['impact'], ('Dampak', 'neutral'))
    return (f'<div class="factor"><div class="factor__head">{status_chip(label, status)}'
            f'<span>{f["name"]}</span></div><div class="factor__body">{f["description"]}</div></div>')


if narrative_result and narrative_result.get('direction') != 'UNKNOWN':
    st.subheader("Mengapa harga diprediksi bergerak")

    direction = narrative_result['direction']
    pct = narrative_result.get('pct_change', 0)
    dir_label, dir_status = DIRECTION_STATUS.get(direction, DIRECTION_STATUS['STABIL'])
    st.markdown(f"""
        <div class="panel direction">
            {status_chip(f"{dir_label} ({pct:+.1f}%)", dir_status)}
            <div class="direction__summary">{narrative_result.get('summary', '')}</div>
        </div>
    """, unsafe_allow_html=True)

    # Faktor berdampak tinggi tampil langsung, sisanya di expander
    factors = narrative_result.get('factors', [])
    if factors:
        high_factors = [f for f in factors if f['impact'] == 'high']
        other_factors = [f for f in factors if f['impact'] != 'high']

        if high_factors:
            st.markdown("".join(_factor_html(f) for f in high_factors), unsafe_allow_html=True)

        if other_factors:
            with st.expander(f"Lihat {len(other_factors)} faktor lainnya", expanded=False):
                st.markdown("".join(_factor_html(f) for f in other_factors), unsafe_allow_html=True)

    # Full narrative in expander
    with st.expander("Baca narasi analisis lengkap", expanded=False):
        st.markdown(narrative_result.get('narrative', ''))

# --- Charts ---
st.subheader("Grafik dan analisis")
tab1, tab2, tab3, tab4 = st.tabs(["Prakiraan", "Antarprovinsi", "Korelasi dan faktor EWS", "Evaluasi model"])

with tab1:
    if p_forecast is not None:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=current_data['date'].tail(90), y=current_data['price'].tail(90),
            mode='lines', name='Harga aktual (90 hari)',
            line=dict(color=CHART['actual'], width=2.5)
        ))

        future_data = p_forecast[p_forecast['ds'] > current_data['date'].max()]

        # Confidence band
        fig.add_trace(go.Scatter(
            x=pd.concat([future_data['ds'], future_data['ds'][::-1]]),
            y=pd.concat([future_data['yhat_upper'], future_data['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(168,106,16,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Interval 90% Prophet'
        ))

        fig.add_trace(go.Scatter(
            x=future_data['ds'], y=future_data['yhat'],
            mode='lines', name='Prakiraan Prophet',
            line=dict(color=CHART['forecast'], width=2, dash='dot')
        ))

        if predicted_price is not None:
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(forecast_date)], y=[predicted_price],
                mode='markers', name=f'Target {model_choice}',
                marker=dict(color=CHART['target'], size=13, symbol='diamond')
            ))

            # Confidence range for target
            if pred_lower and pred_upper:
                fig.add_trace(go.Scatter(
                    x=[pd.Timestamp(forecast_date)]*2,
                    y=[pred_lower, pred_upper],
                    mode='lines', name='Rentang prediksi target',
                    line=dict(color=CHART['target'], width=3),
                ))

        apply_theme_to_plotly(
            fig,
            height=500, margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(title='Harga (IDR/kg)')
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Grafik prakiraan belum tersedia karena perhitungan model gagal. Lihat pesan di bagian atas halaman.")

with tab2:
    latest_all = df[df['commodity'] == selected_commodity].groupby('province').last().reset_index()
    fig_comp = px.bar(
        latest_all.sort_values('price', ascending=False), x='province', y='price',
        color='price', title=f"Harga {selected_commodity} terakhir per provinsi",
        color_continuous_scale="Viridis",
        labels={'price': 'Harga (IDR/kg)', 'province': ''}
    )
    apply_theme_to_plotly(fig_comp)
    st.plotly_chart(fig_comp, width="stretch")

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Korelasi antarkomoditas**")
        prov_data = df[df['province'] == selected_province].pivot(index='date', columns='commodity', values='price')
        corr = prov_data.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1, title=f"Korelasi harga di {selected_province}")
        apply_theme_to_plotly(fig_corr)
        st.plotly_chart(fig_corr, width="stretch")
    with col_b:
        st.markdown("**Skor faktor EWS**")
        factors = ews_result.get('factors', {})
        factor_names = {
            'price_change': 'Perubahan harga',
            'volatility': 'Volatilitas',
            'seasonal': 'Anomali musiman',
            'cross_region': 'Anomali regional',
            'velocity': 'Kecepatan perubahan',
        }
        if not factors:
            st.info("Skor faktor muncul setelah prakiraan berhasil dihitung.")
        for key, score in factors.items():
            st.markdown(f"{factor_names.get(key, key)}: **{score:.0f}/100**")
            st.progress(min(score / 100, 1.0))

        recommendations = ews_result.get('recommendations', [])
        if recommendations:
            st.divider()
            st.markdown("**Rekomendasi**")
            for rec in recommendations:
                st.markdown(f"- {rec}")

with tab4:
    if metrics is not None:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RMSE", f"{metrics['RMSE']:,.2f}")
        mc2.metric("MAE", f"{metrics['MAE']:,.2f}")
        mc3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")

        mc4, mc5, mc6 = st.columns(3)
        mc4.metric("R²", f"{metrics.get('R²', 0):.4f}")
        mc5.metric("SMAPE", f"{metrics.get('SMAPE (%)', 0):.2f}%")
        mc6.metric("Akurasi arah", f"{metrics.get('Directional Accuracy (%)', 0):.1f}%")

        mape = metrics['MAPE (%)']
        st.success(f"**Kategori (Lewis, 1982): {metrics.get('Kategori MAPE', '-')}**, MAPE {mape:.2f}%")
        st.caption("Metrik dihitung dengan protokol rolling-origin (split 80/20, horizon 30 hari) yang sama "
                   "dengan Laboratorium Model. Akurasi arah relatif terhadap harga di titik asal.")
    else:
        st.info("Metrik evaluasi belum tersedia karena perhitungan model gagal. Coba model lain di sidebar.")

st.divider()
render_footer("Model: Prophet, BiLSTM, TFT", "Data: PIHPS Bank Indonesia", "© 2026 Fahmi Prasanda")
