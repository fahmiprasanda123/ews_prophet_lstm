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
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'province': ['Aceh', 'Aceh', 'Aceh'],
        'commodity': ['Beras', 'Beras', 'Beras'],
        'price': [10000, 11000, 12000]
    })
    
    # Setup mock Prophet with realistic return values
    model_instance = MagicMock()
    future_df = pd.DataFrame({'ds': pd.to_datetime(['2024-01-04'])})
    model_instance.make_future_dataframe.return_value = future_df
    model_instance.predict.return_value = pd.DataFrame({
        'ds': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
        'yhat': [10000, 11000, 12000, 12500],
        'yhat_lower': [9500, 10500, 11500, 12000],
        'yhat_upper': [10500, 11500, 12500, 13000],
        'trend': [10000, 10500, 11000, 11500],
    })
    mock_prophet.return_value = model_instance
    
    forecaster = FoodPriceProphet(df)
    forecast = forecaster.train_and_forecast('Aceh', 'Beras', periods=1)
    
    # Verify Prophet was constructed with expected parameters
    assert mock_prophet.called
    prophet_kwargs = mock_prophet.call_args[1]
    assert prophet_kwargs['yearly_seasonality'] is True
    assert prophet_kwargs['daily_seasonality'] is False
    
    # Verify model.fit was called with Prophet-formatted data (ds, y columns)
    assert model_instance.fit.called
    fit_data = model_instance.fit.call_args[0][0]
    assert 'ds' in fit_data.columns, "Prophet data should have 'ds' column"
    assert 'y' in fit_data.columns, "Prophet data should have 'y' column"
    assert len(fit_data) == 3, "Should fit on all 3 data points"
    
    # Verify forecast has expected columns
    assert isinstance(forecast, pd.DataFrame)
    assert 'yhat' in forecast.columns
    assert model_instance.predict.called

@pytest.mark.skipif(Prophet is None, reason="prophet not installed")
def test_prophet_split_data():
    """Test that split_data produces chronological train/test split."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'province': ['Aceh']*5,
        'commodity': ['Beras']*5,
        'price': [10000, 11000, 12000, 13000, 14000]
    })
    
    forecaster = FoodPriceProphet(df)
    p_df = forecaster.prepare_data('Aceh', 'Beras')
    train_df, test_df = forecaster.split_data(p_df, test_size=0.4)
    
    # 60% train, 40% test of 5 points → 3 train, 2 test
    assert len(train_df) == 3
    assert len(test_df) == 2
    
    # Train dates should come before test dates (chronological)
    assert train_df['ds'].max() < test_df['ds'].min()
