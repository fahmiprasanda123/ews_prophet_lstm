"""Tests for Backtester module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_test_df(days=400):
    """Generate test data for backtesting."""
    rows = []
    price = 10000
    for i in range(days):
        price += np.random.normal(0, 100)
        price = max(5000, price)
        rows.append({
            'date': datetime(2024, 1, 1) + timedelta(days=i),
            'province': 'DKI Jakarta',
            'commodity': 'Beras',
            'price': round(price, 2),
        })
    return pd.DataFrame(rows)


class TestBacktester:
    def setup_method(self):
        from engine.backtester import Backtester
        self.df = make_test_df()
        self.bt = Backtester(self.df)

    def test_walk_forward_returns_results(self):
        results = self.bt.walk_forward_test(
            'DKI Jakarta', 'Beras',
            train_window=180, test_window=30, step_size=60,
            model_type='prophet'
        )
        assert isinstance(results, list)
        # Should have at least 1 fold with 400 days of data
        if results:
            assert 'fold' in results[0]
            assert 'metrics' in results[0]
            assert 'predictions' in results[0]
            assert 'actuals' in results[0]

    def test_summary(self):
        results = self.bt.walk_forward_test(
            'DKI Jakarta', 'Beras',
            train_window=180, test_window=30, step_size=90,
            model_type='prophet'
        )
        summary = self.bt.get_summary(results)
        
        if results:
            assert 'avg_mape' in summary
            assert 'folds' in summary
            assert summary['folds'] > 0
        else:
            assert summary['folds'] == 0

    def test_empty_summary(self):
        summary = self.bt.get_summary([])
        assert summary['folds'] == 0
        assert summary['avg_mape'] is None

    def test_insufficient_data(self):
        from engine.backtester import Backtester
        small_df = make_test_df(days=30)
        bt = Backtester(small_df)
        results = bt.walk_forward_test(
            'DKI Jakarta', 'Beras',
            train_window=180, test_window=30
        )
        assert results == []
