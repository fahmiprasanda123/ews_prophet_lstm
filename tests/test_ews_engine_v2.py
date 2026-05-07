"""Tests for EWS Engine v2."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_test_df(days=180, provinces=None, commodities=None, base_price=10000):
    """Generate test DataFrame with realistic price patterns."""
    provinces = provinces or ['DKI Jakarta', 'Jawa Barat']
    commodities = commodities or ['Beras', 'Cabai Merah']
    
    rows = []
    for prov in provinces:
        for comm in commodities:
            price = base_price + np.random.randint(-2000, 2000)
            for i in range(days):
                # Add some trend and noise
                price += np.random.normal(0, price * 0.01)
                price = max(price * 0.5, price)  # Floor
                rows.append({
                    'date': datetime(2025, 1, 1) + timedelta(days=i),
                    'province': prov,
                    'commodity': comm,
                    'price': round(price, 2),
                })
    
    return pd.DataFrame(rows)


class TestEWSEngineV2:
    def setup_method(self):
        from engine.ews_engine_v2 import EWSEngineV2
        self.df = make_test_df()
        self.ews = EWSEngineV2(self.df)

    def test_composite_score_returns_valid_result(self):
        current = self.df[
            (self.df['province'] == 'DKI Jakarta') & 
            (self.df['commodity'] == 'Beras')
        ]['price'].iloc[-1]
        predicted = current * 1.15  # 15% increase

        result = self.ews.calculate_composite_score('DKI Jakarta', 'Beras', predicted)
        
        assert 'score' in result
        assert 'level' in result
        assert 'factors' in result
        assert 'recommendations' in result
        assert 0 <= result['score'] <= 100
        assert result['level'] in ['Danger', 'Alert', 'Watch', 'Normal', 'Unknown']

    def test_danger_level_on_high_increase(self):
        current = self.df[
            (self.df['province'] == 'DKI Jakarta') & 
            (self.df['commodity'] == 'Beras')
        ]['price'].iloc[-1]
        predicted = current * 1.50  # 50% increase

        result = self.ews.calculate_composite_score('DKI Jakarta', 'Beras', predicted)
        assert result['level'] in ['Danger', 'Alert']

    def test_normal_level_on_stable_price(self):
        current = self.df[
            (self.df['province'] == 'DKI Jakarta') & 
            (self.df['commodity'] == 'Beras')
        ]['price'].iloc[-1]
        predicted = current * 1.01  # 1% increase

        result = self.ews.calculate_composite_score('DKI Jakarta', 'Beras', predicted)
        assert result['level'] in ['Normal', 'Watch']

    def test_factors_all_present(self):
        current = self.df[
            (self.df['province'] == 'DKI Jakarta') & 
            (self.df['commodity'] == 'Beras')
        ]['price'].iloc[-1]

        result = self.ews.calculate_composite_score('DKI Jakarta', 'Beras', current * 1.1)
        
        expected_factors = ['price_change', 'volatility', 'seasonal', 'cross_region', 'velocity']
        for f in expected_factors:
            assert f in result['factors']

    def test_backward_compat_warning_level(self):
        level, msg = self.ews.calculate_warning_level(10000, 12500)
        assert level in ['Danger', 'Alert', 'Normal', 'Normal (Downward)']
        assert isinstance(msg, str)

    def test_status_color(self):
        assert self.ews.get_status_color('Danger') == '#FF4B4B'
        assert self.ews.get_status_color('Normal') == '#00CC96'

    def test_empty_data_fallback(self):
        from engine.ews_engine_v2 import EWSEngineV2
        ews = EWSEngineV2(pd.DataFrame(columns=['date', 'province', 'commodity', 'price']))
        result = ews.calculate_composite_score('X', 'Y', 10000)
        assert result['level'] == 'Unknown'
