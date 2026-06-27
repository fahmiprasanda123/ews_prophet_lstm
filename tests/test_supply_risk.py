"""Tests for Supply Risk Scorer."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.supply_risk import SupplyRiskScorer


def make_test_df(days=90, provinces=None, commodities=None, base_price=10000, trend=0):
    """Generate test DataFrame with controllable price patterns."""
    provinces = provinces or ['DKI Jakarta', 'Jawa Barat']
    commodities = commodities or ['Beras', 'Cabai Merah']
    
    rows = []
    for prov in provinces:
        for comm in commodities:
            price = base_price
            for i in range(days):
                price += trend + np.random.normal(0, price * 0.005)
                price = max(1000, price)
                rows.append({
                    'date': datetime(2025, 1, 1) + timedelta(days=i),
                    'province': prov,
                    'commodity': comm,
                    'price': round(price, 2),
                })
    return pd.DataFrame(rows)


class TestSupplyRiskScorer:
    def setup_method(self):
        self.df = make_test_df()
        self.scorer = SupplyRiskScorer(self.df)

    def test_returns_valid_result(self):
        result = self.scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        
        assert 'score' in result
        assert 'factors' in result
        assert 'trend_direction' in result
        assert 'description' in result
        assert 0 <= result['score'] <= 100

    def test_all_four_factors_present(self):
        result = self.scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        expected = ['trend_momentum', 'multi_commodity', 'price_disparity', 'spike_pattern']
        for f in expected:
            assert f in result['factors'], f"Missing factor: {f}"

    def test_factor_weights_sum_to_one(self):
        total = sum(SupplyRiskScorer.FACTOR_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_high_risk_on_rapid_increase(self):
        """Rapidly rising prices should produce higher risk scores."""
        df = make_test_df(days=90, trend=200)  # strong upward trend
        scorer = SupplyRiskScorer(df)
        result = scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        assert result['score'] > 30  # Should be elevated

    def test_stable_prices_low_risk(self):
        """Very stable prices should produce low risk scores."""
        rows = []
        for i in range(90):
            for prov in ['DKI Jakarta', 'Jawa Barat']:
                for comm in ['Beras', 'Cabai Merah']:
                    rows.append({
                        'date': datetime(2025, 1, 1) + timedelta(days=i),
                        'province': prov,
                        'commodity': comm,
                        'price': 10000.0,  # perfectly stable
                    })
        df = pd.DataFrame(rows)
        scorer = SupplyRiskScorer(df)
        result = scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        assert result['score'] < 40

    def test_trend_direction_labels(self):
        result = self.scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        valid_trends = ['↑ Upward', '↓ Downward', '→ Stable', '— Insufficient data']
        assert result['trend_direction'] in valid_trends

    def test_description_matches_score(self):
        result = self.scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        score = result['score']
        desc = result['description']
        if score >= 70:
            assert 'TINGGI' in desc
        elif score >= 45:
            assert 'SEDANG' in desc
        elif score >= 25:
            assert 'RENDAH' in desc
        else:
            assert 'MINIMAL' in desc

    def test_insufficient_data(self):
        """Less than 30 days should use fallback scores."""
        df = make_test_df(days=10)
        scorer = SupplyRiskScorer(df)
        result = scorer.calculate_risk_score('DKI Jakarta', 'Beras')
        assert 'score' in result  # Should not crash
