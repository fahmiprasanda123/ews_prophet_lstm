"""
Page 4: Reports — PDF and Excel Export.
"""
import streamlit as st
import pandas as pd
import datetime
import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import get_store

st.set_page_config(page_title="Reports | Agri-AI EWS", page_icon="📋", layout="wide")

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
st.sidebar.title("📋 Reports")
st.sidebar.markdown("---")
rpt_province = st.sidebar.selectbox("Provinsi", sorted(df['province'].unique()), key="rpt_prov", index=min(10, len(df['province'].unique())-1))
rpt_commodity = st.sidebar.selectbox("Komoditas", sorted(df['commodity'].unique()), key="rpt_comm", index=0)

st.title("📋 Report Generator")
st.markdown("### Generate laporan analisis untuk download")

st.markdown(f"""
<div style="
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
">
    <strong>Target Laporan:</strong> {rpt_commodity} di {rpt_province}
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📄 PDF Report", "📊 Excel Export", "📈 Data Preview"])

# --- Tab 1: PDF Report ---
with tab1:
    st.markdown("### 📄 Laporan PDF Lengkap")
    st.markdown("Generate laporan profesional dengan analisis EWS, prediksi harga, dan rekomendasi tindakan.")

    if st.button("🚀 Generate PDF Report", key="gen_pdf"):
        with st.spinner("Generating report..."):
            try:
                from engine.ews_engine_v2 import EWSEngineV2
                from engine.supply_risk import SupplyRiskScorer
                from models.prophet_forecast import FoodPriceProphet
                from models.evaluation import calculate_metrics
                from prophet import Prophet

                # Get current price
                series = df[(df['province'] == rpt_province) & (df['commodity'] == rpt_commodity)].sort_values('date')
                current_price = series['price'].iloc[-1]

                # Run forecast
                fp = FoodPriceProphet(df)
                p_df = fp.prepare_data(rpt_province, rpt_commodity)
                train_df, test_df = fp.split_data(p_df, test_size=0.2)
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(train_df)
                eval_pred = m.predict(test_df[['ds']])
                metrics = calculate_metrics(test_df['y'].values, eval_pred['yhat'].values, "Prophet")

                forecast = fp.train_and_forecast(rpt_province, rpt_commodity, periods=30)
                predicted_price = forecast['yhat'].iloc[-1]

                # EWS
                ews = EWSEngineV2(df)
                ews_result = ews.calculate_composite_score(rpt_province, rpt_commodity, predicted_price)

                # Supply Risk
                scorer = SupplyRiskScorer(df)
                supply_risk = scorer.calculate_risk_score(rpt_province, rpt_commodity)

                # Generate PDF
                from report.pdf_generator import EWSReportGenerator
                gen = EWSReportGenerator()
                pdf_bytes = gen.generate_report(
                    province=rpt_province,
                    commodity=rpt_commodity,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    ews_result=ews_result,
                    supply_risk=supply_risk,
                    metrics=metrics,
                    forecast_date=datetime.datetime.now().strftime('%d %B %Y'),
                )

                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"EWS_Report_{rpt_province}_{rpt_commodity}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )

                # Preview
                st.markdown("---")
                st.markdown("### Preview")
                pc1, pc2 = st.columns(2)
                pc1.metric("EWS Level", ews_result['level'])
                pc2.metric("EWS Score", f"{ews_result['score']}/100")

                pc3, pc4 = st.columns(2)
                pc3.metric("Harga Saat Ini", f"IDR {current_price:,.0f}")
                pc4.metric("Harga Prediksi", f"IDR {predicted_price:,.0f}")

            except ImportError as e:
                st.error(f"❌ Dependency belum terinstall: {e}")
                st.info("Install dengan: `pip install fpdf2`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Tab 2: Excel Export ---
with tab2:
    st.markdown("### 📊 Export Data ke Excel")

    date_range = st.date_input(
        "Rentang Tanggal",
        value=(df['date'].max() - pd.Timedelta(days=90), df['date'].max()),
        key="excel_dates"
    )

    export_scope = st.radio("Scope Data", ["Komoditas & Provinsi Terpilih", "Semua Komoditas (Provinsi Terpilih)", "Semua Data"])

    if st.button("📥 Generate Excel", key="gen_excel"):
        with st.spinner("Preparing Excel file..."):
            try:
                if export_scope == "Komoditas & Provinsi Terpilih":
                    export_df = df[
                        (df['province'] == rpt_province) & 
                        (df['commodity'] == rpt_commodity)
                    ].copy()
                elif export_scope == "Semua Komoditas (Provinsi Terpilih)":
                    export_df = df[df['province'] == rpt_province].copy()
                else:
                    export_df = df.copy()

                if len(date_range) == 2:
                    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                    export_df = export_df[(export_df['date'] >= start) & (export_df['date'] <= end)]

                export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')

                # Create Excel with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Raw data
                    export_df.to_excel(writer, sheet_name='Data Harga', index=False)

                    # Pivot table
                    if 'commodity' in export_df.columns and len(export_df['commodity'].unique()) > 1:
                        pivot = export_df.pivot_table(
                            index='date', columns='commodity', values='price', aggfunc='mean'
                        )
                        pivot.to_excel(writer, sheet_name='Pivot Komoditas')

                    # Summary statistics
                    summary = export_df.groupby('commodity')['price'].agg(
                        ['mean', 'std', 'min', 'max', 'count']
                    ).round(2)
                    summary.columns = ['Rata-rata', 'Std Dev', 'Minimum', 'Maximum', 'Jumlah Data']
                    summary.to_excel(writer, sheet_name='Statistik')

                output.seek(0)

                st.success(f"✅ Excel ready! ({len(export_df)} records)")
                st.download_button(
                    label="⬇️ Download Excel",
                    data=output.getvalue(),
                    file_name=f"EWS_Data_{rpt_province}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except ImportError:
                st.error("❌ openpyxl belum terinstall. Install dengan: `pip install openpyxl`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# --- Tab 3: Data Preview ---
with tab3:
    st.markdown("### 📈 Preview Data")

    preview_data = df[
        (df['province'] == rpt_province) & 
        (df['commodity'] == rpt_commodity)
    ].sort_values('date', ascending=False).head(100)

    preview_data_display = preview_data.copy()
    preview_data_display['date'] = preview_data_display['date'].dt.strftime('%Y-%m-%d')
    preview_data_display['price'] = preview_data_display['price'].apply(lambda x: f"IDR {x:,.0f}")

    st.dataframe(preview_data_display, use_container_width=True, height=500)

    # Quick stats
    cs1, cs2, cs3, cs4 = st.columns(4)
    prices = preview_data['price']
    cs1.metric("Terakhir", f"IDR {prices.iloc[0]:,.0f}")
    cs2.metric("Rata-rata (100d)", f"IDR {prices.mean():,.0f}")
    cs3.metric("Min", f"IDR {prices.min():,.0f}")
    cs4.metric("Max", f"IDR {prices.max():,.0f}")
