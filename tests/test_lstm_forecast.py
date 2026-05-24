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

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lstm_predict_shape():
    forecaster = LSTMForecaster(seq_length=30)
    last_30 = np.random.randint(10000, 12000, 30)
    
    # Need to fit scaler first
    forecaster.scaler.fit(last_30.reshape(-1, 1))
    
    # Model belum dilatih — harus raise error
    with pytest.raises(RuntimeError, match="belum dilatih"):
        forecaster.predict(last_30)

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lstm_prepare_and_split_no_data_leakage():
    """Verify that prepare_and_split fits scaler only on training data."""
    np.random.seed(42)
    n = 200
    seq_length = 10
    # Create data where the LAST 20% of targets have much higher prices
    # With n=200, seq=10: total_seq=190, split_idx=152, train_end_raw=162
    # So training scaler covers raw[0:162], test TARGETS are raw[162:199]
    prices = np.concatenate([
        np.random.randint(10000, 11000, 162),   # Training region  
        np.random.randint(50000, 60000, 38),     # Test target region (very different)
    ])
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=n, freq='D'),
        'province': ['Aceh']*n,
        'commodity': ['Beras']*n,
        'price': prices
    })
    
    forecaster = LSTMForecaster(seq_length=seq_length)
    X_train, X_test, y_train, y_test = forecaster.prepare_and_split(
        df, 'Aceh', 'Beras', test_size=0.2
    )
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    
    # Scaler was fit on raw[0:162] which is in the 10000-11000 range
    # So data_max should be ~11000, NOT 50000-60000
    scaler_max = forecaster.scaler.data_max_[0]
    assert scaler_max < 15000, (
        f"Scaler max {scaler_max} suggests data leakage "
        f"(test target data included in scaler fit)"
    )
    
    # Verify that test targets exist and are different from training
    # After inverse_transform, test targets should be in the 50000+ range
    y_test_actual = forecaster.scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    assert y_test_actual.max() > 15000, "Test targets should contain out-of-training-range values"

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_lstm_train_and_predict():
    """End-to-end test: prepare, split, train, predict."""
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=80, freq='D'),
        'province': ['Aceh']*80,
        'commodity': ['Beras']*80,
        'price': np.random.randint(10000, 12000, 80)
    })
    
    forecaster = LSTMForecaster(seq_length=10)
    X_train, X_test, y_train, y_test = forecaster.prepare_and_split(
        df, 'Aceh', 'Beras', test_size=0.2
    )
    
    forecaster.train_single_series(X_train, y_train, epochs=2)
    assert forecaster._is_trained is True
    
    # Single prediction
    last_10 = np.random.randint(10000, 12000, 10)
    prediction = forecaster.predict(last_10)
    assert prediction.shape == (1, 1)
    
    # Multi-step prediction
    multi = forecaster.predict_multi_step(last_10, steps=5)
    assert multi.shape == (5,)

