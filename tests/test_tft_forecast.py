"""Tests for TFT Forecaster."""
import pytest
import pandas as pd
import numpy as np
from models.tft_forecast import TFTForecaster, TFT_AVAILABLE


class TestTFTForecaster:
    def test_init_defaults(self):
        forecaster = TFTForecaster()
        assert forecaster.max_prediction_length == 30
        assert forecaster.max_encoder_length == 90
        assert forecaster.model is None

    def test_is_available_property(self):
        forecaster = TFTForecaster()
        assert forecaster.is_available == TFT_AVAILABLE

    @pytest.mark.skipif(not TFT_AVAILABLE, reason="pytorch-forecasting not installed")
    def test_prepare_dataset_filters_province(self):
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=200, freq='D').tolist() * 2,
            'province': ['DKI Jakarta'] * 200 + ['Jawa Barat'] * 200,
            'commodity': ['Beras'] * 400,
            'price': np.random.randint(10000, 12000, 400),
        })
        forecaster = TFTForecaster()
        dataset = forecaster.prepare_dataset(df, province='DKI Jakarta', commodity='Beras')
        assert dataset is not None

    @pytest.mark.skipif(TFT_AVAILABLE, reason="Only test when TFT not installed")
    def test_prepare_dataset_raises_without_lib(self):
        forecaster = TFTForecaster()
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'province': ['DKI Jakarta'] * 100,
            'commodity': ['Beras'] * 100,
            'price': [10000] * 100,
        })
        with pytest.raises(RuntimeError, match="pytorch-forecasting"):
            forecaster.prepare_dataset(df)

    def test_custom_init_params(self):
        forecaster = TFTForecaster(max_prediction_length=14, max_encoder_length=60)
        assert forecaster.max_prediction_length == 14
        assert forecaster.max_encoder_length == 60


def _tiny_series(start='2024-01-01', days=150, seed=0):
    idx = pd.date_range(start, periods=days, freq='D')
    rng = np.random.default_rng(seed)
    return pd.Series(10000 + np.cumsum(rng.normal(0, 50, days)), index=idx)


@pytest.mark.skipif(not TFT_AVAILABLE, reason="pytorch-forecasting not installed")
class TestTFTShortWindow:
    def test_unseen_month_in_prediction_window(self):
        """Jendela latih pendek tanpa bulan September tidak membuat prediksi gagal."""
        y = _tiny_series(start='2024-03-01', days=200)  # latih s.d. Agustus, prediksi September
        tft = TFTForecaster(max_prediction_length=30, max_encoder_length=30)
        frame = tft.build_frame(y, 'Jawa Timur', 'Beras')
        ds = tft.build_training_dataset(frame, training_cutoff=len(y) - 31)
        assert set(frame['month'].iloc[-30:]) - set(frame['month'].iloc[:-30])
        tft.train(ds, max_epochs=1, batch_size=32, hidden_size=8, attention_head_size=1, quiet=True)
        res = tft.predict(frame, ds)
        assert len(res['mean']) == 30

    def test_full_year_keeps_default_encoders(self):
        from models.tft_forecast import _time_encoders
        frame = TFTForecaster.build_frame(_tiny_series(days=400), 'Jawa Timur', 'Beras')
        assert _time_encoders(frame) == {}
        assert set(_time_encoders(frame.iloc[:100])) == {'month'}
