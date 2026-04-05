import pytest
import pandas as pd
import numpy as np
from models.lstm_forecast import LSTMForecaster, torch

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lstm_prepare_data():
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=40, freq='D'),
        'province': ['Aceh']*40,
        'commodity': ['Beras']*40,
        'price': np.random.randint(10000, 12000, 40)
    })
    
    forecaster = LSTMForecaster(seq_length=30)
    X, y = forecaster.prepare_data(df, 'Aceh', 'Beras')
    
    # seq_length = 30, total 40 points -> 10 sequences
    assert X.shape[0] == 10
    assert X.shape[1] == 30 # seq_length
    assert X.shape[2] == 1 # input_size
    assert y.shape[0] == 10

from models.lstm_forecast import LSTMForecaster, torch

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lstm_predict_shape():
    forecaster = LSTMForecaster(seq_length=30)
    last_30 = np.random.randint(10000, 12000, 30)
    
    # Need to fit scaler first
    forecaster.scaler.fit(last_30.reshape(-1, 1))
    
    prediction = forecaster.predict(last_30)
    assert prediction.shape == (1, 1)
