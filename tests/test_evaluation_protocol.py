"""Regresi untuk protokol evaluasi tesis dan prakiraan ke depan."""
import numpy as np
import pandas as pd
import pytest

from models.evaluation_protocol import (
    DEFAULT_CONFIG, derive_covariates, exante_covariates, interval_coverage, make_windows, prepare_series, run_sma,
)


def _series(days=60):
    idx = pd.date_range('2024-01-01', periods=days, freq='D')
    return pd.Series(np.linspace(100, 160, days), index=idx, name='price')


class TestShortInputs:
    def test_sma_origin_shorter_than_window(self):
        y = _series()
        out = run_sma(y, make_windows(10, 20, 10), window=30)
        assert out['pred'].notna().all()
        assert out['pred'].iloc[0] == pytest.approx(y.iloc[:10].mean())

    def test_sma_without_history_raises(self):
        with pytest.raises(ValueError, match="SMA"):
            run_sma(_series(), [{'origin': 0, 'keep_from': 0, 'end': 5}])

    def test_exante_without_history_raises(self):
        y = _series()
        cov = pd.DataFrame({'rainfall_30d': 1.0, 'temperature_30d': 27.0, 'enso_avail': 0.5}, index=y.index)
        with pytest.raises(ValueError, match="riwayat kovariat"):
            exante_covariates(cov, y.index[0], y.index[:5])


class TestEnsoLag:
    def test_lag_matches_thesis(self):
        # Jeda ini tercatat di thesis_outputs/konfigurasi.json; mengubahnya mengubah Bab IV.
        assert DEFAULT_CONFIG['enso_lag_days'] == 30

    def test_value_used_only_after_lag(self):
        idx = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        oni = pd.Series(0.0, index=idx)
        oni[idx >= '2023-06-15'] = 1.0
        raw = pd.DataFrame({'rainfall_mm': 1.0, 'temperature_c': 27.0, 'enso_index': oni})
        cov = derive_covariates(raw, idx)
        first = cov.index[cov['enso_avail'] == 1.0][0]
        assert first == pd.Timestamp('2023-06-15') + pd.Timedelta(days=DEFAULT_CONFIG['enso_lag_days'])


class TestIntervalCoverage:
    def test_nan_bounds_skipped(self):
        y = _series(10)
        df = pd.DataFrame({'pred': y.values, 'lower': np.nan, 'upper': np.nan}, index=y.index)
        assert interval_coverage({'TFT': df}, y).empty

    def test_partial_nan_bounds_use_valid_days(self):
        y = _series(10)
        df = pd.DataFrame({'pred': y.values, 'lower': y.values - 1, 'upper': y.values + 1}, index=y.index)
        df.iloc[:5, 1:] = np.nan
        out = interval_coverage({'TFT': df}, y)
        assert out['Cakupan aktual (%)'].iloc[0] == 100.0


def test_prepare_series_ffill_then_interpolate():
    """Celah 10 hari: 7 hari pertama forward-fill, 3 sisanya interpolasi (sesuai docstring)."""
    dates = list(pd.date_range('2024-01-01', periods=5)) + list(pd.date_range('2024-01-16', periods=5))
    df = pd.DataFrame({'date': dates, 'province': 'P', 'commodity': 'K',
                       'price': [100.0] * 5 + [140.0] * 5})
    y, info = prepare_series(df, 'P', 'K', ffill_limit=7)
    assert info['n_ffilled'] == 7 and info['n_interpolated'] == 3
    assert (y['2024-01-06':'2024-01-12'] == 100).all()
    np.testing.assert_allclose(y['2024-01-13':'2024-01-15'].values, [110, 120, 130])


def test_forecast_path_reports_failed_tft(monkeypatch):
    """TFT gagal -> dicatat, ensemble hanya dari model yang berhasil."""
    import models.tft_forecast as tft_mod
    from models.forecast_ahead import forecast_path

    def boom(*args, **kwargs):
        raise RuntimeError('uji gagal')
    monkeypatch.setattr(tft_mod.TFTForecaster, 'train', boom)
    rng = np.random.default_rng(0)
    dates = pd.date_range('2023-01-01', periods=400, freq='D')
    df = pd.DataFrame({'date': dates, 'province': 'P', 'commodity': 'K',
                       'price': 10000 + np.cumsum(rng.normal(0, 30, len(dates)))})
    cfg = {'lstm': {'epochs': 1, 'mc_samples': 3, 'hidden_size': 8}, 'tft': {'max_epochs': 1},
           'prophet': {'holidays_country': None}}
    res = forecast_path(df, 'P', 'K', horizon=7, config=cfg,
                        weights={'Prophet': 0.4, 'BiLSTM': 0.3, 'TFT': 0.3}, best_single='Prophet')
    assert 'TFT' not in res['path'].columns
    assert res['ensemble_models'] == ['Prophet', 'BiLSTM']
    assert res['bound_models'] == ['Prophet', 'BiLSTM']
    assert any('TFT gagal' in n for n in res['notes'])
    assert any('tanpa TFT' in n for n in res['notes'])
    assert res['path']['Smart Ensemble'].notna().all() and res['path']['Batas bawah'].notna().all()
    assert res['enso_lag_days'] == DEFAULT_CONFIG['enso_lag_days']
