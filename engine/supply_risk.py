"""
Supply Risk Scoring module for Agri-AI EWS.
Calculates a composite supply risk score based on multiple market indicators.
"""
import numpy as np
import pandas as pd


class SupplyRiskScorer:
    """Computes supply risk score (0-100) from price data patterns."""

    FACTOR_WEIGHTS = {
        'trend_momentum': 0.30,
        'multi_commodity': 0.25,
        'price_disparity': 0.25,
        'spike_pattern': 0.20,
    }

    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def calculate_risk_score(self, province, commodity):
        """Calculate composite supply risk score.
        
        Returns:
            dict with 'score' (0-100), 'factors', 'trend_direction', 'description'
        """
        factors = {
            'trend_momentum': self._score_trend_momentum(province, commodity),
            'multi_commodity': self._score_multi_commodity(province),
            'price_disparity': self._score_price_disparity(commodity),
            'spike_pattern': self._score_spike_pattern(province, commodity),
        }

        composite = np.clip(
            sum(factors[k] * self.FACTOR_WEIGHTS[k] for k in factors), 0, 100
        )

        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date')
        
        recent_change = 0
        trend = "— Insufficient data"
        if len(series) >= 7:
            recent_change = (series['price'].iloc[-1] - series['price'].iloc[-7]) / series['price'].iloc[-7] * 100
            trend = "↑ Upward" if recent_change > 2 else ("↓ Downward" if recent_change < -2 else "→ Stable")

        return {
            'score': round(composite, 1),
            'factors': {k: round(v, 1) for k, v in factors.items()},
            'trend_direction': trend,
            'weekly_change_pct': round(recent_change, 2),
            'description': self._get_description(composite),
        }

    def _score_trend_momentum(self, province, commodity):
        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date')['price']
        if len(series) < 30:
            return 30
        current = series.iloc[-1]
        velocities = []
        for w in [7, 14, 30]:
            past = series.iloc[-w]
            if past > 0:
                velocities.append((current - past) / past * 100)
        if not velocities:
            return 30
        if len(velocities) >= 2 and velocities[0] > velocities[-1] > 0:
            return min(100, velocities[0] * 8)
        elif velocities[0] > 0:
            return min(80, velocities[0] * 5)
        return max(0, 30 + velocities[0] * 2)

    def _score_multi_commodity(self, province, window=14):
        prov_data = self.df[self.df['province'] == province]
        if prov_data.empty:
            return 30
        start = prov_data['date'].max() - pd.Timedelta(days=window)
        recent = prov_data[prov_data['date'] >= start]
        rising = total = 0
        for comm in recent['commodity'].unique():
            cd = recent[recent['commodity'] == comm].sort_values('date')
            if len(cd) >= 2:
                total += 1
                if cd['price'].iloc[0] > 0 and (cd['price'].iloc[-1] - cd['price'].iloc[0]) / cd['price'].iloc[0] > 0.02:
                    rising += 1
        return min(100, (rising / total * 120)) if total > 0 else 30

    def _score_price_disparity(self, commodity):
        latest_date = self.df[self.df['commodity'] == commodity]['date'].max()
        if pd.isna(latest_date):
            return 30
        prices = self.df[(self.df['commodity'] == commodity) & (self.df['date'] == latest_date)]['price']
        if len(prices) < 3:
            return 30
        cv = prices.std() / prices.mean() * 100 if prices.mean() > 0 else 0
        return min(100, cv * 3.3)

    def _score_spike_pattern(self, province, commodity):
        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date')
        if len(series) < 14:
            return 20
        prices = series['price'].values
        recent_14d = prices[-14:]
        recent_change = (recent_14d[-1] - recent_14d[0]) / recent_14d[0] if recent_14d[0] > 0 else 0
        if recent_change > 0.075:
            return min(100, recent_change / 0.15 * 80)
        elif recent_change > 0:
            return min(60, recent_change / 0.15 * 40)
        return 10

    @staticmethod
    def _get_description(score):
        if score >= 70:
            return "Risiko pasokan TINGGI. Intervensi mungkin diperlukan."
        elif score >= 45:
            return "Risiko pasokan SEDANG. Monitor pergerakan harga."
        elif score >= 25:
            return "Risiko pasokan RENDAH. Pasar relatif stabil."
        return "Risiko pasokan MINIMAL. Kondisi pasokan normal."
