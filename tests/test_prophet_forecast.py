import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from models.prophet_forecast import FoodPriceProphet, Prophet

@pytest.mark.skipif(Prophet is None, reason="prophet not installed")
def test_prophet_prepare_data():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'province': ['Aceh', 'Aceh'],
        'commodity': ['Beras', 'Beras'],
        'price': [10000, 11000]
    })
    
    forecaster = FoodPriceProphet(df)
    p_df = forecaster.prepare_data('Aceh', 'Beras')
    
    assert 'ds' in p_df.columns
    assert 'y' in p_df.columns
    assert p_df['ds'].iloc[0] == pd.Timestamp('2024-01-01')
    assert p_df['y'].iloc[0] == 10000

@pytest.mark.skipif(Prophet is None, reason="prophet not installed")
@patch("models.prophet_forecast.Prophet")
def test_prophet_train_and_forecast_mock(mock_prophet):
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'province': ['Aceh', 'Aceh'],
        'commodity': ['Beras', 'Beras'],
        'price': [10000, 11000]
    })
    
    # Setup mock Prophet
    model_instance = MagicMock()
    model_instance.make_future_dataframe.return_value = pd.DataFrame()
    model_instance.predict.return_value = pd.DataFrame({'ds': [], 'yhat': []})
    mock_prophet.return_value = model_instance
    
    forecaster = FoodPriceProphet(df)
    forecast = forecaster.train_and_forecast('Aceh', 'Beras', periods=1)
    
    assert mock_prophet.called
    assert model_instance.fit.called
    assert model_instance.predict.called
    assert isinstance(forecast, pd.DataFrame)
