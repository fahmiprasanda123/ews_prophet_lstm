"""Tests for Price Narrative Analyzer."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engine.price_narrative import PriceNarrativeAnalyzer


def make_narrative_df(days=180, province='DKI Jakarta', commodity='Beras', base_price=10000):
    """Generate test DataFrame for narrative analysis."""
    rows = []
    price = base_price
    for i in range(days):
        price += np.random.normal(0, price * 0.005)
        price = max(1000, price)
        rows.append({
            'date': datetime(2025, 1, 1) + timedelta(days=i),
            'province': province,
            'commodity': commodity,
            'price': round(price, 2),
        })
    return pd.DataFrame(rows)


class TestPriceNarrativeAnalyzer:
    def setup_method(self):
        np.random.seed(42)
        self.df = make_narrative_df()
        self.analyzer = PriceNarrativeAnalyzer(self.df)

    def test_generate_returns_valid_structure(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.10, '2025-08-01'
        )
        assert 'direction' in result
        assert 'pct_change' in result
        assert 'summary' in result
        assert 'factors' in result
        assert 'narrative' in result

    def test_direction_naik(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.15
        )
        assert result['direction'] == 'NAIK'
        assert result['pct_change'] > 0

    def test_direction_turun(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 0.85
        )
        assert result['direction'] == 'TURUN'
        assert result['pct_change'] < 0

    def test_direction_stabil(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.005
        )
        assert result['direction'] == 'STABIL'

    def test_factors_have_required_keys(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.10, '2025-08-01'
        )
        for factor in result['factors']:
            assert 'name' in factor
            assert 'impact' in factor
            assert 'description' in factor

    def test_factors_sorted_by_impact(self):
        """Factors should be sorted: high > medium > low."""
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.30, '2025-08-01'
        )
        impacts = [f['impact'] for f in result['factors']]
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        for i in range(len(impacts) - 1):
            assert impact_order.get(impacts[i], 3) <= impact_order.get(impacts[i+1], 3)

    def test_pct_change_calculation(self):
        current = self.df['price'].iloc[-1]
        predicted = current * 1.20  # 20% increase
        result = self.analyzer.generate_narrative('DKI Jakarta', 'Beras', predicted)
        assert abs(result['pct_change'] - 20.0) < 1.0

    def test_insufficient_data_fallback(self):
        """Less than 14 data points should return fallback narrative."""
        small_df = make_narrative_df(days=5)
        analyzer = PriceNarrativeAnalyzer(small_df)
        result = analyzer.generate_narrative('DKI Jakarta', 'Beras', 12000)
        assert 'direction' in result
        assert 'summary' in result

    def test_empty_data_fallback(self):
        empty_df = pd.DataFrame(columns=['date', 'province', 'commodity', 'price'])
        analyzer = PriceNarrativeAnalyzer(empty_df)
        result = analyzer.generate_narrative('DKI Jakarta', 'Beras', 12000)
        assert 'direction' in result

    def test_all_commodities_in_knowledge_base(self):
        """All 10 strategic commodities should be in the KB."""
        expected = [
            'Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam',
            'Bawang Merah', 'Bawang Putih', 'Cabai Merah', 'Cabai Rawit',
            'Minyak Goreng', 'Gula Pasir'
        ]
        for comm in expected:
            assert comm in PriceNarrativeAnalyzer.COMMODITY_KB, f"Missing KB entry: {comm}"

    def test_commodity_kb_structure(self):
        """Each KB entry should have required fields."""
        required_keys = [
            'harvest_months', 'scarce_months', 'weather_sensitivity',
            'holiday_sensitivity', 'harvest_description', 'scarce_description',
            'weather_description', 'holiday_description'
        ]
        for comm, data in PriceNarrativeAnalyzer.COMMODITY_KB.items():
            for key in required_keys:
                assert key in data, f"Missing '{key}' in KB for {comm}"

    def test_narrative_is_string(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.10, '2025-08-01'
        )
        assert isinstance(result['narrative'], str)
        assert len(result['narrative']) > 50  # Should be a substantive narrative

    def test_summary_is_string(self):
        current = self.df['price'].iloc[-1]
        result = self.analyzer.generate_narrative(
            'DKI Jakarta', 'Beras', current * 1.10
        )
        assert isinstance(result['summary'], str)
        assert len(result['summary']) > 10
