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


class TestBacktesterBaselines:
    def setup_method(self):
        from engine.backtester import Backtester
        np.random.seed(0)
        self.bt = Backtester(make_test_df(days=270))

    @pytest.mark.parametrize('model_type', ['naive', 'sma', 'arima'])
    def test_conventional_backtests_return_folds(self, model_type):
        results = self.bt.walk_forward_test('DKI Jakarta', 'Beras', train_window=180, test_window=30,
                                            step_size=30, model_type=model_type)
        assert len(results) == 3
        assert all(len(r['predictions']) == 30 for r in results)

    def test_naive_fallback_is_labelled(self):
        results = self.bt.walk_forward_test('DKI Jakarta', 'Beras', train_window=180, test_window=30,
                                            step_size=30, model_type='naive')
        assert 'naive nilai terakhir' in results[0]['note']
        assert np.allclose(results[0]['predictions'], results[0]['predictions'][0])

    def test_naive_with_fitting_period_has_no_note(self):
        results = self.bt.walk_forward_test('DKI Jakarta', 'Beras', train_window=180, test_window=30,
                                            step_size=30, model_type='naive',
                                            model_params={'seasonal_period': 7})
        assert 'note' not in results[0]
        preds = np.array(results[0]['predictions'])
        np.testing.assert_array_equal(preds[:7], preds[7:14])  # pola mingguan berulang

    def test_lstm_fold_reproducible(self):
        import pandas as pd
        series = self.bt.df.sort_values('date').reset_index(drop=True)
        train, test = series.iloc[:120], series.iloc[120:130]
        params = {'epochs': 1, 'hidden_size': 8, 'seq_length': 14}
        a = self.bt._run_lstm_fold(train, test, params)
        b = self.bt._run_lstm_fold(train, test, params)
        np.testing.assert_array_equal(a, b)

    def test_tft_fold_with_unseen_month(self, tmp_path, monkeypatch):
        from models.tft_forecast import TFT_AVAILABLE
        if not TFT_AVAILABLE:
            pytest.skip("pytorch-forecasting not installed")
        monkeypatch.chdir(tmp_path)  # lightning_logs tidak ditulis ke repositori
        series = self.bt.df.sort_values('date').reset_index(drop=True)
        train, test = series.iloc[:150], series.iloc[150:180]  # Jan–Mei latih, Mei–Jun uji
        preds = self.bt._run_tft_fold(train, test, 'DKI Jakarta', 'Beras',
                                      {'tft_max_epochs': 1, 'tft_batch_size': 32})
        assert preds is not None and len(preds) == 30
