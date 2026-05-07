"""
PDF Report Generator for Agri-AI EWS.
Uses fpdf2 to create professional forecast reports.
"""
import io
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


class EWSReportGenerator:
    """Generates PDF reports for Agri-AI EWS."""

    def __init__(self):
        if not FPDF_AVAILABLE:
            raise ImportError("fpdf2 is required. Install with: pip install fpdf2")

    def generate_report(self, province, commodity, current_price, predicted_price,
                        ews_result, supply_risk, metrics=None, 
                        forecast_date=None):
        """Generate a PDF report.
        
        Returns:
            bytes: PDF content as bytes for download.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Cover Page ---
        pdf.add_page()
        pdf.set_fill_color(14, 17, 23)
        pdf.rect(0, 0, 210, 297, 'F')

        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 28)
        pdf.ln(60)
        pdf.cell(0, 15, "Agri-AI Early Warning System", ln=True, align="C")
        
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(79, 172, 254)
        pdf.cell(0, 10, "Laporan Analisis Harga Pangan", ln=True, align="C")

        pdf.ln(20)
        pdf.set_text_color(200, 200, 200)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Komoditas: {commodity}", ln=True, align="C")
        pdf.cell(0, 8, f"Provinsi: {province}", ln=True, align="C")
        pdf.cell(0, 8, f"Tanggal Laporan: {datetime.now().strftime('%d %B %Y')}", ln=True, align="C")
        if forecast_date:
            pdf.cell(0, 8, f"Target Prediksi: {forecast_date}", ln=True, align="C")

        pdf.ln(40)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "Powered by Prophet + BiLSTM + Temporal Fusion Transformer", ln=True, align="C")
        pdf.cell(0, 6, "Fahmi Prasanda", ln=True, align="C")

        # --- Executive Summary ---
        pdf.add_page()
        pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Executive Summary", ln=True)
        pdf.ln(5)

        # EWS Status Box
        ews_level = ews_result.get('level', 'Unknown')
        ews_score = ews_result.get('score', 0)
        ews_color = {
            'Danger': (255, 75, 75), 'Alert': (255, 165, 0),
            'Watch': (255, 215, 0), 'Normal': (0, 204, 150),
        }.get(ews_level, (100, 100, 100))

        pdf.set_fill_color(*ews_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 12, f"  Status EWS: {ews_level.upper()} (Score: {ews_score}/100)", ln=True, fill=True)

        pdf.set_text_color(30, 30, 30)
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, ews_result.get('message', ''), ln=True)

        # Price Summary Table
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Ringkasan Harga", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(90, 8, "Indikator", 1, 0, "L", True)
        pdf.cell(90, 8, "Nilai", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 10)
        
        pct_change = ews_result.get('pct_change', 0)
        rows = [
            ("Harga Pasar Terakhir", f"IDR {current_price:,.0f}/kg"),
            ("Harga Prediksi", f"IDR {predicted_price:,.0f}/kg"),
            ("Perubahan Harga", f"{pct_change:+.2f}%"),
            ("Supply Risk Score", f"{supply_risk.get('score', 'N/A')}/100"),
            ("Tren Pasokan", supply_risk.get('trend_direction', 'N/A')),
        ]
        for label, value in rows:
            pdf.cell(90, 7, f"  {label}", 1, 0, "L")
            pdf.cell(90, 7, value, 1, 1, "C")

        # --- EWS Factor Analysis ---
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Analisis Faktor EWS", ln=True)
        pdf.ln(3)

        factors = ews_result.get('factors', {})
        factor_labels = {
            'price_change': 'Perubahan Harga',
            'volatility': 'Volatilitas Pasar',
            'seasonal': 'Anomali Musiman',
            'cross_region': 'Anomali Regional',
            'velocity': 'Kecepatan Perubahan',
        }

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(80, 8, "Faktor", 1, 0, "L", True)
        pdf.cell(40, 8, "Score", 1, 0, "C", True)
        pdf.cell(60, 8, "Level", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 10)
        for key, score in factors.items():
            label = factor_labels.get(key, key)
            level = "Tinggi" if score > 60 else ("Sedang" if score > 30 else "Rendah")
            pdf.cell(80, 7, f"  {label}", 1, 0, "L")
            pdf.cell(40, 7, f"{score:.1f}", 1, 0, "C")
            pdf.cell(60, 7, level, 1, 1, "C")

        # --- Model Metrics ---
        if metrics:
            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Performa Model AI", ln=True)
            pdf.ln(3)

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(60, 8, "Metrik", 1, 0, "L", True)
            pdf.cell(60, 8, "Nilai", 1, 0, "C", True)
            pdf.cell(60, 8, "Interpretasi", 1, 1, "C", True)

            pdf.set_font("Helvetica", "", 10)
            metric_rows = [
                ("RMSE", f"{metrics.get('RMSE', 0):,.2f}", "Deviasi rata-rata (IDR)"),
                ("MAE", f"{metrics.get('MAE', 0):,.2f}", "Error absolut rata-rata"),
                ("MAPE", f"{metrics.get('MAPE (%)', 0):.2f}%", 
                 "Sangat Baik" if metrics.get('MAPE (%)', 100) < 10 else "Baik"),
                ("R\u00b2", f"{metrics.get('R\u00b2', 0):.4f}", "Proporsi varians dijelaskan"),
            ]
            for label, value, interp in metric_rows:
                pdf.cell(60, 7, f"  {label}", 1, 0, "L")
                pdf.cell(60, 7, value, 1, 0, "C")
                pdf.cell(60, 7, interp, 1, 1, "C")

        # --- Recommendations ---
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Rekomendasi Tindakan", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 10)
        for rec in ews_result.get('recommendations', []):
            pdf.cell(5, 7, "", 0, 0)
            pdf.multi_cell(0, 7, f"  {rec}")

        # --- Footer ---
        pdf.ln(20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 5, "Agri-AI EWS v2.0 | Fahmi Prasanda", ln=True)
        pdf.cell(0, 5, "Disclaimer: Prediksi ini bersifat estimasi dan tidak menjamin akurasi absolut.", ln=True)

        return bytes(pdf.output())

    def generate_excel_data(self, df, province, commodity, forecast_data=None):
        """Generate Excel-ready dict of DataFrames.
        
        Returns:
            dict of sheet_name → DataFrame
        """
        sheets = {}

        # Historical data
        historical = df[
            (df['province'] == province) & (df['commodity'] == commodity)
        ][['date', 'price']].copy()
        historical.columns = ['Tanggal', 'Harga (IDR/kg)']
        sheets['Data Historis'] = historical

        # Forecast data
        if forecast_data is not None:
            sheets['Prediksi'] = forecast_data

        # Summary stats
        if not historical.empty:
            summary = {
                'Metrik': ['Harga Terakhir', 'Rata-rata 30 Hari', 'Min 30 Hari',
                          'Max 30 Hari', 'Std Dev 30 Hari'],
                'Nilai': [
                    historical['Harga (IDR/kg)'].iloc[-1],
                    historical['Harga (IDR/kg)'].tail(30).mean(),
                    historical['Harga (IDR/kg)'].tail(30).min(),
                    historical['Harga (IDR/kg)'].tail(30).max(),
                    historical['Harga (IDR/kg)'].tail(30).std(),
                ]
            }
            import pandas as pd
            sheets['Ringkasan'] = pd.DataFrame(summary)

        return sheets
