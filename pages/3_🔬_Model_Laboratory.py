"""
Page 3: Model Laboratory — Model Comparison, Backtesting, and Analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_store

st.set_page_config(page_title="Model Lab | Agri-AI EWS", page_icon="🔬", layout="wide")

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

df = load_data()
if df.empty:
    st.error("❌ Data tidak tersedia.")
    st.stop()

# Sidebar
st.sidebar.title("🔬 Model Laboratory")
render_theme_toggle()
st.sidebar.markdown("---")
lab_province = st.sidebar.selectbox("Provinsi", sorted(df['province'].unique()), key="lab_prov", index=min(10, len(df['province'].unique())-1))
lab_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), key="lab_comm", index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Parameters")

with st.sidebar.expander("🔮 Prophet Config", expanded=False):
    p_cps = st.slider("Changepoint Prior Scale", 0.001, 0.5, st.session_state.model_params['changepoint_prior_scale'], format="%.3f")
    p_yearly = st.checkbox("Yearly Seasonality", st.session_state.model_params['yearly_seasonality'])
    p_weekly = st.checkbox("Weekly Seasonality", st.session_state.model_params['weekly_seasonality'])

with st.sidebar.expander("🧠 LSTM Config", expanded=False):
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

model_params = st.session_state.model_params

st.title("🔬 Model Laboratory")
st.markdown("### Bandingkan performa model AI dan jalankan backtesting")

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔄 Backtesting", "📈 Variable Importance"])

# --- Tab 1: Model Comparison ---
with tab1:
    st.markdown(f"**{lab_commodity}** di **{lab_province}** — split 80/20 + rolling-origin (horizon 30 hari)")
    st.caption(
        "Protokol sama dengan skrip tesis (scripts/thesis_outputs.py): setiap model hanya memakai data "
        "sebelum titik asal, semua prediksi disejajarkan per tanggal, dan bobot Smart Ensemble dicari "
        "dengan grid search pada periode validasi (20% terakhir data latih)."
    )
    use_cov = st.checkbox("Gunakan kovariat iklim nyata (Open-Meteo & NOAA)", value=True, key="lab_cov")

    if st.button("🚀 Jalankan Perbandingan Model", key="run_compare"):
        from models.evaluation_protocol import evaluate_series, load_climate_covariates, prepare_series

        lab_config = {
            'prophet': {'changepoint_prior_scale': float(model_params['changepoint_prior_scale']),
                        'yearly_seasonality': bool(model_params['yearly_seasonality']),
                        'weekly_seasonality': bool(model_params['weekly_seasonality'])},
            'lstm': {'epochs': int(model_params['epochs']), 'hidden_size': int(model_params['hidden_size']),
                     'seq_length': int(model_params['seq_length'])},
            'tft': {'max_epochs': int(model_params['tft_max_epochs']),
                    'batch_size': int(model_params['tft_batch_size'])},
        }
        cov_raw = None
        if use_cov:
            try:
                y_tmp, _ = prepare_series(df, lab_province, lab_commodity)
                with st.spinner("Mengambil kovariat iklim (Open-Meteo & NOAA)..."):
                    cov_raw = load_climate_covariates(lab_province, y_tmp.index.min(), y_tmp.index.max(),
                                                      strict=True)
            except Exception as e:
                st.warning(f"Kovariat iklim nyata tidak tersedia ({e}). Perbandingan dijalankan TANPA kovariat.")
        bar = st.progress(0.0, text="Menyiapkan data...")
        st.session_state['lab_result'] = evaluate_series(
            df, lab_province, lab_commodity, config=lab_config, covariates_raw=cov_raw,
            progress=lambda msg, frac: bar.progress(min(max(float(frac), 0.0), 1.0), text=msg),
        )

    result = st.session_state.get('lab_result')
    if result is not None and (result['province'], result['commodity']) == (lab_province, lab_commodity):
        from models.evaluation_protocol import predictions_frame

        sp = result['split']
        st.info(
            f"Latih {sp['n_train']} hari (s.d. {sp['train_end']}) · Uji {sp['n_test']} hari "
            f"({sp['test_start']} s.d. {sp['test_end']}) · {sp['n_windows_test']} titik asal · "
            f"Kovariat iklim: {'ya' if result['covariates_used'] else 'tidak'}"
        )
        for note in result['notes']:
            st.warning(note)

        st.markdown("### 📋 Tabel Perbandingan Metrik")
        metrics_df = result['metrics']
        display_cols = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)', 'R²',
                        'Directional Accuracy (%)', 'Kategori MAPE']
        st.dataframe(
            metrics_df[display_cols].style.format({
                'RMSE': '{:,.0f}', 'MAE': '{:,.0f}', 'MAPE (%)': '{:.2f}', 'SMAPE (%)': '{:.2f}',
                'R²': '{:.3f}', 'Directional Accuracy (%)': '{:.1f}',
            }).highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)'], color='#00CC96')
              .highlight_max(subset=['R²', 'Directional Accuracy (%)'], color='#00CC96'),
            use_container_width=True,
        )
        best = metrics_df.loc[metrics_df['MAPE (%)'].idxmin()]
        st.success(f"🏆 MAPE terendah: **{best['Model']}** ({best['MAPE (%)']:.2f}%, kategori {best['Kategori MAPE']})")
        st.caption("Directional Accuracy dihitung terhadap harga terakhir di titik asal. "
                   "Kategori MAPE mengikuti Lewis (1982).")

        if result['ensemble']:
            ens_info = result['ensemble']
            wcols = st.columns(len(ens_info['weights']))
            for col, (name, weight) in zip(wcols, ens_info['weights'].items()):
                col.metric(f"Bobot {name}", f"{weight * 100:.0f}%")
            st.caption(
                f"Bobot hasil grid search (langkah 5%) pada validasi {sp['n_val']} hari; "
                f"fallback ke {ens_info['best_single']} aktif pada {ens_info['fallback_days']} hari uji "
                f"({ens_info['fallback_rate']:.1f}%)."
            )

        st.markdown("### 📉 Prediksi vs Aktual (Test Set)")
        pf = predictions_frame(result)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pf.index, y=pf['aktual'], mode='lines', name='Aktual',
                                 line=dict(color=theme_color('plotly_actual_line'), width=3)))
        colors = {'Naive Seasonal': '#888888', 'SMA-30': '#A0A0A0', 'ARIMA(5,1,0)': '#C0C0C0',
                  'Prophet': '#4facfe', 'BiLSTM': '#FFA500', 'TFT': '#FF4B4B', 'Smart Ensemble': '#00CC96'}
        for name, color in colors.items():
            if name in pf.columns:
                fig.add_trace(go.Scatter(x=pf.index, y=pf[name], mode='lines', name=name,
                                         line=dict(color=color, width=2, dash='dot')))
        for origin in sorted(set(pf['titik_asal'])):
            fig.add_vline(x=origin, line_width=0.5, line_color='rgba(128,128,128,0.35)')
        apply_theme_to_plotly(fig, height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title='Harga (IDR/kg)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ⏱️ MAPE menurut Horizon Prediksi")
        st.dataframe(result['horizon_mape'].style.format(precision=2), use_container_width=True)
        if not result['intervals'].empty:
            st.markdown("### 🎯 Cakupan Interval Ketidakpastian")
            st.dataframe(result['intervals'].style.format(precision=2), use_container_width=True)
        st.download_button(
            "⬇️ Unduh prediksi per tanggal (CSV)", pf.to_csv().encode('utf-8'),
            file_name=f"prediksi_{lab_commodity}_{lab_province}.csv".replace(' ', '_'), mime='text/csv',
        )
    else:
        st.info("👆 Klik tombol di atas untuk menjalankan perbandingan model.")

# --- Tab 2: Backtesting ---
with tab2:
    st.markdown(f"### Walk-Forward Backtesting — {lab_commodity} di {lab_province}")
    st.markdown("Menguji performa model pada data historis dengan sliding window.")

    bc1, bc2, bc3 = st.columns(3)
    train_window = bc1.number_input("Training Window (hari)", 90, 365, 180)
    test_window = bc2.number_input("Test Window (hari)", 7, 60, 30)
    step_size = bc3.number_input("Step Size (hari)", 7, 60, 30)

    bt_model = st.selectbox("Model untuk Backtest", ["naive", "sma", "arima", "prophet", "lstm", "tft", "ensemble"])

    if st.button("🔄 Jalankan Backtesting", key="run_bt"):
        from engine.backtester import Backtester
        bt = Backtester(df)

        st.info(f"⚙️ Backtesting with: {bt_model.upper()} | Params: {model_params}")
        with st.spinner("Backtesting sedang berjalan... Ini bisa memakan waktu beberapa menit."):
            results = bt.walk_forward_test(
                lab_province, lab_commodity,
                train_window=train_window, test_window=test_window,
                step_size=step_size, model_type=bt_model,
                model_params=model_params
            )

        if results:
            summary = bt.get_summary(results)

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Folds", summary['folds'])
            sc2.metric("Avg MAPE", f"{summary['avg_mape']:.2f}%")
            sc3.metric("Avg RMSE", f"{summary['avg_rmse']:,.0f}")
            sc4.metric("Avg R²", f"{summary['avg_r2']:.4f}")

            # MAPE per fold chart
            fold_mapes = [r['metrics']['MAPE (%)'] for r in results]
            fold_labels = [f"Fold {r['fold']}\n{r['test_start']}" for r in results]

            fig_bt = go.Figure(go.Bar(
                x=fold_labels, y=fold_mapes,
                marker_color=['#00CC96' if m < 10 else '#FFA500' if m < 20 else '#FF4B4B' for m in fold_mapes],
                text=[f"{m:.1f}%" for m in fold_mapes], textposition='auto',
            ))
            fig_bt.add_hline(y=summary['avg_mape'], line_dash="dash", line_color=theme_color('plotly_hline'),
                            annotation_text=f"Avg: {summary['avg_mape']:.1f}%",
                            annotation_font_color=theme_color('text_primary'))
            apply_theme_to_plotly(
                fig_bt, height=400,
                title="MAPE per Fold",
            )
            fig_bt.update_yaxes(title='MAPE (%)')
            st.plotly_chart(fig_bt, use_container_width=True)

            # Actual vs Predicted overlay for best fold
            best = results[summary['best_fold']]
            fig_best = go.Figure()
            fig_best.add_trace(go.Scatter(y=best['actuals'], name='Actual', line=dict(color=theme_color('plotly_actual_line'), width=2)))
            fig_best.add_trace(go.Scatter(y=best['predictions'], name='Predicted', line=dict(color='#4facfe', width=2, dash='dot')))
            apply_theme_to_plotly(
                fig_best, height=350,
                title=f"Best Fold ({best['test_start']} → {best['test_end']}) — MAPE: {best['metrics']['MAPE (%)']:.2f}%",
            )
            st.plotly_chart(fig_best, use_container_width=True)

            # EWS accuracy test
            st.markdown("### 🔔 Akurasi Deteksi EWS")
            st.caption("Skenario prediksi sempurna (hindsight): harga aktual masa depan dipakai sebagai "
                       "prediksi. Angka ini batas atas kemampuan logika skor EWS, bukan kinerja model prediksi.")
            with st.spinner("Testing EWS alert accuracy..."):
                from engine.ews_engine_v2 import EWSEngineV2
                ews = EWSEngineV2(df)
                ews_acc = bt.test_ews_accuracy(lab_province, lab_commodity, ews)

            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Detection Rate", f"{ews_acc['detection_rate']:.1f}%")
            ec2.metric("Total Spikes Found", ews_acc['total_spikes'])
            ec3.metric("Avg Lead Time", f"{ews_acc['avg_lead_time']:.0f} hari")

            if ews_acc.get('events'):
                st.dataframe(pd.DataFrame(ews_acc['events']), use_container_width=True)
        else:
            st.warning("Tidak cukup data untuk backtesting dengan parameter ini.")

# --- Tab 3: Variable Importance ---
with tab3:
    st.markdown("### 📊 Analisis Variabel Penting")
    st.markdown("Faktor-faktor yang paling berpengaruh terhadap prediksi harga.")

    # Since TFT variable importance requires a trained model,
    # show feature analysis from data instead
    series = df[(df['province'] == lab_province) & (df['commodity'] == lab_commodity)].sort_values('date')

    if len(series) > 60:
        # Create lag features and compute correlation
        price = series['price'].values
        features = {
            'Lag 1d': np.corrcoef(price[1:], price[:-1])[0, 1],
            'Lag 7d': np.corrcoef(price[7:], price[:-7])[0, 1] if len(price) > 7 else 0,
            'Lag 14d': np.corrcoef(price[14:], price[:-14])[0, 1] if len(price) > 14 else 0,
            'Lag 30d': np.corrcoef(price[30:], price[:-30])[0, 1] if len(price) > 30 else 0,
            'Month': abs(series['date'].dt.month.corr(series['price'])),
            'Day of Week': abs(series['date'].dt.dayofweek.corr(series['price'])),
        }

        # Cross commodity correlations
        prov_pivot = df[df['province'] == lab_province].pivot(index='date', columns='commodity', values='price')
        if lab_commodity in prov_pivot.columns:
            for other_comm in prov_pivot.columns:
                if other_comm != lab_commodity:
                    corr_val = prov_pivot[lab_commodity].corr(prov_pivot[other_comm])
                    if not np.isnan(corr_val):
                        features[f'Corr: {other_comm}'] = abs(corr_val)

        # Sort and display
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)

        fig_imp = go.Figure(go.Bar(
            y=[f[0] for f in sorted_features],
            x=[abs(f[1]) for f in sorted_features],
            orientation='h',
            marker_color=[
                '#4facfe' if f[0].startswith('Lag') else
                '#FFA500' if f[0].startswith('Corr') else '#00CC96'
                for f in sorted_features
            ],
            text=[f"{abs(f[1]):.3f}" for f in sorted_features],
            textposition='auto',
        ))
        apply_theme_to_plotly(
            fig_imp, height=max(300, len(sorted_features) * 30),
            title="Feature Importance (Absolute Correlation)",
        )
        fig_imp.update_xaxes(title='|Correlation|')
        fig_imp.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        > **Interpretasi**: 
        > - 🔵 **Lag features**: Autokorelasi harga — seberapa tergantung harga hari ini pada harga sebelumnya
        > - 🟠 **Cross-commodity**: Korelasi dengan komoditas lain — mengindikasikan hubungan supply chain
        > - 🟢 **Temporal**: Pengaruh waktu (bulan, hari) — mengindikasikan pola musiman
        """)
    else:
        st.warning("Tidak cukup data untuk analisis variabel.")
