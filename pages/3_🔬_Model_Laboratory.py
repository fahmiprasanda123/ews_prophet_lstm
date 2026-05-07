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
st.sidebar.markdown("---")
lab_province = st.sidebar.selectbox("Provinsi", sorted(df['province'].unique()), key="lab_prov", index=min(10, len(df['province'].unique())-1))
lab_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), key="lab_comm", index=0)

st.title("🔬 Model Laboratory")
st.markdown("### Bandingkan performa model AI dan jalankan backtesting")

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔄 Backtesting", "📈 Variable Importance"])

# --- Tab 1: Model Comparison ---
with tab1:
    st.markdown(f"**{lab_commodity}** di **{lab_province}** — 80/20 Time-Series Split")

    if st.button("🚀 Jalankan Perbandingan Model", key="run_compare"):
        from models.evaluation import calculate_metrics, compare_models

        all_metrics = []
        predictions = {}

        # Prophet
        with st.spinner("Training Prophet..."):
            try:
                from models.prophet_forecast import FoodPriceProphet
                from prophet import Prophet
                fp = FoodPriceProphet(df)
                p_df = fp.prepare_data(lab_province, lab_commodity)
                train_df, test_df = fp.split_data(p_df, test_size=0.2)
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(train_df)
                pred = m.predict(test_df[['ds']])
                metrics = calculate_metrics(test_df['y'].values, pred['yhat'].values, "Prophet")
                all_metrics.append(metrics)
                predictions['Prophet'] = {'dates': test_df['ds'].values, 'pred': pred['yhat'].values, 'actual': test_df['y'].values}
            except Exception as e:
                st.warning(f"Prophet error: {e}")

        # LSTM
        with st.spinner("Training BiLSTM..."):
            try:
                import torch
                from models.lstm_forecast import LSTMForecaster
                lf = LSTMForecaster(seq_length=30)
                X, y = lf.prepare_data(df, lab_province, lab_commodity)
                Xtr, Xte, ytr, yte = lf.split_data(X, y, test_size=0.2)
                lf.train_single_series(Xtr, ytr, epochs=10)
                lf.model.eval()
                with torch.no_grad():
                    yp = lf.model(Xte)
                    y_pred_lstm = lf.scaler.inverse_transform(yp.numpy().reshape(-1, 1)).flatten()
                    y_true_lstm = lf.scaler.inverse_transform(yte.numpy().reshape(-1, 1)).flatten()
                    metrics = calculate_metrics(y_true_lstm, y_pred_lstm, "BiLSTM")
                    all_metrics.append(metrics)
                    predictions['BiLSTM'] = {'pred': y_pred_lstm, 'actual': y_true_lstm}
            except Exception as e:
                st.warning(f"LSTM error: {e}")

        # TFT (check availability)
        try:
            from models.tft_forecast import get_tft_forecaster
            tft = get_tft_forecaster()
            if tft.is_available:
                with st.spinner("Training TFT..."):
                    try:
                        dataset, data = tft.prepare_dataset(df, lab_province, lab_commodity)
                        if dataset is not None:
                            tft.train(dataset, max_epochs=5, batch_size=32)
                            tft_pred = tft.predict(data, dataset)
                            if tft_pred is not None:
                                all_metrics.append({
                                    "Model": "TFT",
                                    "RMSE": 0, "MAE": 0, "MAPE (%)": 0,
                                    "R²": 0, "SMAPE (%)": 0, "Directional Accuracy (%)": 0,
                                })
                    except Exception as e:
                        st.info(f"TFT: {e}")
            else:
                st.info("ℹ️ TFT tidak tersedia (pytorch-forecasting belum diinstall).")
        except Exception:
            st.info("ℹ️ TFT module not available.")

        if all_metrics:
            # Metrics comparison table
            st.markdown("### 📋 Tabel Perbandingan Metrik")
            metrics_df = pd.DataFrame(all_metrics)
            display_cols = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'SMAPE (%)', 'Directional Accuracy (%)']
            available_cols = [c for c in display_cols if c in metrics_df.columns]
            st.dataframe(metrics_df[available_cols].style.highlight_min(
                subset=[c for c in ['RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)'] if c in available_cols],
                color='#00CC96'
            ).highlight_max(
                subset=[c for c in ['R²', 'Directional Accuracy (%)'] if c in available_cols],
                color='#00CC96'
            ), use_container_width=True)

            comparison = compare_models(all_metrics)
            st.success(f"🏆 Best Model: **{comparison['best_model']}** (MAPE: {comparison['best_mape']:.2f}%)")

            # Overlay chart
            if 'Prophet' in predictions:
                st.markdown("### 📉 Prediksi vs Aktual (Test Set)")
                fig = go.Figure()

                if 'dates' in predictions['Prophet']:
                    x_axis = predictions['Prophet']['dates']
                else:
                    x_axis = list(range(len(predictions['Prophet']['actual'])))

                fig.add_trace(go.Scatter(
                    x=x_axis, y=predictions['Prophet']['actual'],
                    mode='lines', name='Actual', line=dict(color='white', width=3)
                ))
                
                colors = {'Prophet': '#4facfe', 'BiLSTM': '#FFA500', 'TFT': '#FF4B4B'}
                for model_name, pred_data in predictions.items():
                    pred_vals = pred_data['pred']
                    x = x_axis[:len(pred_vals)] if len(pred_vals) <= len(x_axis) else list(range(len(pred_vals)))
                    fig.add_trace(go.Scatter(
                        x=x, y=pred_vals,
                        mode='lines', name=model_name,
                        line=dict(color=colors.get(model_name, '#FF4B4B'), width=2, dash='dot')
                    ))

                fig.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', height=450,
                    yaxis=dict(title='Harga (IDR/kg)', gridcolor='rgba(255,255,255,0.1)'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Radar chart for metrics comparison
            if len(all_metrics) >= 2:
                st.markdown("### 🕸️ Radar Perbandingan")
                categories = ['RMSE\n(lower=better)', 'MAE\n(lower=better)', 'MAPE\n(lower=better)', 
                             'R²\n(higher=better)', 'Dir. Accuracy\n(higher=better)']
                fig_radar = go.Figure()
                for m in all_metrics:
                    # Normalize to 0-1 scale (invert for "lower is better" metrics)
                    max_rmse = max(mx['RMSE'] for mx in all_metrics) or 1
                    max_mae = max(mx['MAE'] for mx in all_metrics) or 1
                    max_mape = max(mx['MAPE (%)'] for mx in all_metrics) or 1
                    vals = [
                        1 - m['RMSE'] / max_rmse,
                        1 - m['MAE'] / max_mae,
                        1 - m['MAPE (%)'] / max_mape,
                        max(0, m.get('R²', 0)),
                        m.get('Directional Accuracy (%)', 50) / 100,
                    ]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=categories + [categories[0]],
                        fill='toself', name=m['Model'], opacity=0.6,
                    ))
                fig_radar.update_layout(
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                    polar=dict(bgcolor='rgba(0,0,0,0)'), height=400,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
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

    bt_model = st.selectbox("Model untuk Backtest", ["prophet", "lstm"])

    if st.button("🔄 Jalankan Backtesting", key="run_bt"):
        from engine.backtester import Backtester
        bt = Backtester(df)

        with st.spinner("Backtesting sedang berjalan... Ini bisa memakan waktu beberapa menit."):
            results = bt.walk_forward_test(
                lab_province, lab_commodity,
                train_window=train_window, test_window=test_window,
                step_size=step_size, model_type=bt_model,
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
            fig_bt.add_hline(y=summary['avg_mape'], line_dash="dash", line_color="white",
                            annotation_text=f"Avg: {summary['avg_mape']:.1f}%")
            fig_bt.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=400,
                title="MAPE per Fold", yaxis=dict(title='MAPE (%)'),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Actual vs Predicted overlay for best fold
            best = results[summary['best_fold']]
            fig_best = go.Figure()
            fig_best.add_trace(go.Scatter(y=best['actuals'], name='Actual', line=dict(color='white', width=2)))
            fig_best.add_trace(go.Scatter(y=best['predictions'], name='Predicted', line=dict(color='#4facfe', width=2, dash='dot')))
            fig_best.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=350,
                title=f"Best Fold ({best['test_start']} → {best['test_end']}) — MAPE: {best['metrics']['MAPE (%)']:.2f}%",
            )
            st.plotly_chart(fig_best, use_container_width=True)

            # EWS accuracy test
            st.markdown("### 🔔 Akurasi Deteksi EWS")
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
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=max(300, len(sorted_features) * 30),
            title="Feature Importance (Absolute Correlation)",
            xaxis=dict(title='|Correlation|'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        > **Interpretasi**: 
        > - 🔵 **Lag features**: Autokorelasi harga — seberapa tergantung harga hari ini pada harga sebelumnya
        > - 🟠 **Cross-commodity**: Korelasi dengan komoditas lain — mengindikasikan hubungan supply chain
        > - 🟢 **Temporal**: Pengaruh waktu (bulan, hari) — mengindikasikan pola musiman
        """)
    else:
        st.warning("Tidak cukup data untuk analisis variabel.")
