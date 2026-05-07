"""
EWS Engine v2 — Intelligent Multi-Factor Early Warning System.
Replaces the simple threshold-based engine with a composite scoring model.

Factors:
  1. Predicted price change (%)
  2. Historical volatility (rolling std)
  3. Seasonal adjustment (is the change normal for this month?)
  4. Cross-region anomaly detection
  5. Velocity of change (acceleration)
"""
import pandas as pd
import numpy as np
from datetime import timedelta


class EWSEngineV2:
    """Multi-factor Early Warning System engine."""

    # Weight configuration for composite scoring
    FACTOR_WEIGHTS = {
        'price_change': 0.30,
        'volatility': 0.20,
        'seasonal': 0.20,
        'cross_region': 0.15,
        'velocity': 0.15,
    }

    # Risk level thresholds (composite score 0-100)
    RISK_THRESHOLDS = {
        'Danger': 70,
        'Alert': 45,
        'Watch': 25,
    }

    def __init__(self, df):
        """
        Args:
            df: Full historical DataFrame with date, province, commodity, price.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def calculate_composite_score(self, province, commodity, predicted_price, 
                                   forecast_date=None):
        """Calculate the composite EWS risk score.
        
        Args:
            province: Province name.
            commodity: Commodity name.
            predicted_price: Predicted future price.
            forecast_date: Target forecast date (for seasonal context).
        
        Returns:
            dict with 'score' (0-100), 'level', 'message', 'factors', 'recommendations'
        """
        series = self.df[
            (self.df['province'] == province) & 
            (self.df['commodity'] == commodity)
        ].sort_values('date')

        if series.empty or len(series) < 30:
            return self._fallback_result(predicted_price)

        current_price = series['price'].iloc[-1]
        
        # Calculate individual factors (each returns 0-100)
        factors = {}
        factors['price_change'] = self._score_price_change(current_price, predicted_price)
        factors['volatility'] = self._score_volatility(series)
        factors['seasonal'] = self._score_seasonal_anomaly(series, predicted_price, forecast_date)
        factors['cross_region'] = self._score_cross_region(commodity, current_price, predicted_price)
        factors['velocity'] = self._score_velocity(series, predicted_price)

        # Weighted composite
        composite = sum(
            factors[k] * self.FACTOR_WEIGHTS[k] 
            for k in factors
        )
        composite = np.clip(composite, 0, 100)

        # Determine risk level
        level = self._get_level(composite)
        message = self._get_message(level, factors, current_price, predicted_price)
        recommendations = self._get_recommendations(level, factors)

        return {
            'score': round(composite, 1),
            'level': level,
            'message': message,
            'color': self.get_status_color(level),
            'factors': {k: round(v, 1) for k, v in factors.items()},
            'recommendations': recommendations,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'pct_change': ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0,
        }

    # ------------------------------------------------------------------ #
    # Factor scoring (each returns 0-100)
    # ------------------------------------------------------------------ #
    def _score_price_change(self, current, predicted):
        """Score based on percentage price change. Higher change = higher risk."""
        if current <= 0:
            return 0
        pct = abs(predicted - current) / current * 100
        # Scale: 0% → 0, 10% → 50, 20%+ → 100
        score = min(100, pct * 5)
        # Bonus for increases (more concerning than decreases)
        if predicted > current:
            score = min(100, score * 1.2)
        return score

    def _score_volatility(self, series, window=30):
        """Score based on recent price volatility."""
        if len(series) < window:
            return 30  # Default moderate

        recent = series['price'].tail(window)
        volatility = recent.pct_change().std() * 100
        
        # Scale: 0% std → 0, 3% std → 50, 6%+ std → 100
        return min(100, volatility * 16.67)

    def _score_seasonal_anomaly(self, series, predicted_price, forecast_date=None):
        """Score how much the prediction deviates from seasonal norms."""
        if forecast_date is None:
            target_month = pd.Timestamp.now().month
        else:
            target_month = pd.Timestamp(forecast_date).month

        # Calculate historical monthly average
        series_with_month = series.copy()
        series_with_month['month'] = series_with_month['date'].dt.month
        monthly_avg = series_with_month.groupby('month')['price'].mean()
        monthly_std = series_with_month.groupby('month')['price'].std()

        if target_month not in monthly_avg.index:
            return 30

        expected = monthly_avg[target_month]
        std = monthly_std.get(target_month, expected * 0.1)
        
        if std <= 0:
            std = expected * 0.05

        # Z-score of prediction vs seasonal expectation
        z_score = abs(predicted_price - expected) / std
        
        # Scale: z=0 → 0, z=2 → 50, z=4+ → 100
        return min(100, z_score * 25)

    def _score_cross_region(self, commodity, current_price, predicted_price):
        """Score based on how the province compares to national trend."""
        national = self.df[self.df['commodity'] == commodity]
        if national.empty:
            return 30

        # Get latest national average
        latest_date = national['date'].max()
        national_latest = national[national['date'] == latest_date]['price']
        
        if national_latest.empty:
            return 30

        nat_avg = national_latest.mean()
        nat_std = national_latest.std()

        if nat_std <= 0:
            nat_std = nat_avg * 0.1

        # How far is predicted from national average?
        z_nat = abs(predicted_price - nat_avg) / nat_std
        
        # Also check if current price is already an outlier
        z_current = abs(current_price - nat_avg) / nat_std if nat_std > 0 else 0
        
        # Score: outlier provinces get higher risk
        return min(100, max(z_nat, z_current) * 20)

    def _score_velocity(self, series, predicted_price, windows=[7, 14, 30]):
        """Score based on the acceleration of price changes."""
        if len(series) < max(windows):
            return 30

        current = series['price'].iloc[-1]
        velocities = []

        for w in windows:
            past_price = series['price'].iloc[-w]
            if past_price > 0:
                velocity = (current - past_price) / past_price * 100
                velocities.append(velocity)

        if not velocities:
            return 30

        # Check if velocity is increasing (acceleration)
        avg_velocity = np.mean(velocities)
        
        # Future velocity implied by prediction
        if current > 0:
            future_vel = (predicted_price - current) / current * 100
        else:
            future_vel = 0

        # If both historical and future velocity are positive and increasing = danger
        if avg_velocity > 0 and future_vel > avg_velocity:
            acceleration = future_vel - avg_velocity
            return min(100, acceleration * 10)
        elif future_vel > 0:
            return min(100, future_vel * 5)
        else:
            return max(0, 20 + future_vel * 2)  # Decreasing = lower risk

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_level(self, score):
        if score >= self.RISK_THRESHOLDS['Danger']:
            return 'Danger'
        elif score >= self.RISK_THRESHOLDS['Alert']:
            return 'Alert'
        elif score >= self.RISK_THRESHOLDS['Watch']:
            return 'Watch'
        else:
            return 'Normal'

    def _get_message(self, level, factors, current_price, predicted_price):
        pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        direction = "naik" if pct > 0 else "turun"

        messages = {
            'Danger': f"⚠️ BAHAYA: Harga diprediksi {direction} {abs(pct):.1f}%. Risiko tinggi lonjakan harga akut!",
            'Alert': f"🔸 WASPADA: Harga diprediksi {direction} {abs(pct):.1f}%. Perlu monitoring intensif.",
            'Watch': f"👀 PERHATIAN: Harga diprediksi {direction} {abs(pct):.1f}%. Tren perlu diawasi.",
            'Normal': f"✅ STABIL: Harga diprediksi {direction} {abs(pct):.1f}%. Kondisi pasar normal.",
        }
        return messages.get(level, "Status tidak diketahui.")

    def _get_recommendations(self, level, factors):
        recs = []
        
        if level in ['Danger', 'Alert']:
            recs.append("📌 Aktifkan monitoring harga harian untuk komoditas ini.")
            
            if factors.get('volatility', 0) > 60:
                recs.append("📊 Volatilitas tinggi terdeteksi. Pertimbangkan operasi pasar (OP) untuk stabilisasi.")
            
            if factors.get('cross_region', 0) > 60:
                recs.append("🗺️ Anomali regional terdeteksi. Periksa jalur distribusi antar provinsi.")
            
            if factors.get('velocity', 0) > 60:
                recs.append("🚀 Akselerasi kenaikan harga terdeteksi. Risiko eskalasi dalam 1-2 minggu.")
            
            if factors.get('seasonal', 0) > 60:
                recs.append("📅 Perubahan melebihi pola musiman normal. Kemungkinan ada faktor eksternal.")
        
        elif level == 'Watch':
            recs.append("📋 Pantau perkembangan harga dalam 7 hari ke depan.")
            recs.append("📊 Evaluasi kembali jika tren berlanjut.")
        
        else:
            recs.append("✅ Tidak diperlukan tindakan khusus. Pasar dalam kondisi stabil.")

        return recs

    def _fallback_result(self, predicted_price):
        return {
            'score': 0,
            'level': 'Unknown',
            'message': '⚠️ Data tidak cukup untuk analisis EWS.',
            'color': '#666666',
            'factors': {},
            'recommendations': ['Kumpulkan lebih banyak data historis (minimal 30 hari).'],
            'current_price': 0,
            'predicted_price': predicted_price,
            'pct_change': 0,
        }

    @staticmethod
    def get_status_color(level):
        colors = {
            "Danger": "#FF4B4B",
            "Alert": "#FFA500",
            "Watch": "#FFD700",
            "Normal": "#00CC96",
            "Unknown": "#666666",
        }
        return colors.get(level, "#FFFFFF")

    # ------------------------------------------------------------------ #
    # Backward compatibility with EWSEngine v1
    # ------------------------------------------------------------------ #
    def calculate_warning_level(self, current_price, predicted_price):
        """Backward-compatible method matching EWSEngine v1 interface."""
        pct_change = (predicted_price - current_price) / current_price if current_price > 0 else 0
        
        if pct_change >= 0.20:
            return "Danger", "⚠️ High risk of acute price surge!"
        elif pct_change >= 0.10:
            return "Alert", "🔸 Moderate price increase predicted."
        elif pct_change <= -0.10:
            return "Normal (Downward)", "📉 Price is expected to decrease."
        else:
            return "Normal", "✅ Price is stable."
