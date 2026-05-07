"""
LSTM Forecaster for Agri-AI EWS.
Supports single-step and multi-step forecasting with Bidirectional LSTM.
Includes MC Dropout for uncertainty estimation.
"""
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    class DummyModule:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return None
    nn = type('dummy', (), {'Module': DummyModule, 'LSTM': DummyModule, 'Linear': DummyModule, 'MSELoss': DummyModule, 'Dropout': DummyModule})
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


class PriceLSTM(nn.Module):
    """Bidirectional LSTM for price prediction with MC Dropout."""
    
    def __init__(self, input_size=1, hidden_layer_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_layer_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional doubles the hidden size
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # Take the last time step
        last_out = lstm_out[:, -1]
        dropped = self.dropout(last_out)
        predictions = self.linear(dropped)
        return predictions


class LSTMForecaster:
    """LSTM-based forecaster with multi-step and uncertainty features."""
    
    def __init__(self, seq_length=30, hidden_size=128, num_layers=2, dropout=0.2):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.model = PriceLSTM(
            input_size=1,
            hidden_layer_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def prepare_data(self, df, province, commodity):
        if torch is None:
            raise RuntimeError("PyTorch (torch) is not installed correctly. Forecast cannot be prepared.")
            
        subset = df[(df['province'] == province) & (df['commodity'] == commodity)]['price'].values
        subset = subset.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(subset)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i+self.seq_length])
            y.append(scaled_data[i+self.seq_length])
            
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
        
    def split_data(self, X, y, test_size=0.2):
        """
        Melakukan pemisahan dataset menjadi Training dan Testing secara berurutan (Time-series split).
        """
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def train_single_series(self, X, y, epochs=20, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(X)
            single_loss = loss_fn(y_pred, y)
            single_loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1} loss: {single_loss.item():10.8f}')

    def predict(self, last_sequence):
        """Predict the next single value."""
        if torch is None or self.model is None:
            raise RuntimeError("Model or PyTorch is not available.")
            
        self.model.eval()
        with torch.no_grad():
            scaled_seq = self.scaler.transform(last_sequence.reshape(-1, 1))
            input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)
            prediction = self.model(input_tensor)
            return self.scaler.inverse_transform(prediction.numpy())

    def predict_multi_step(self, last_sequence, steps=30):
        """Multi-step recursive forecast.
        
        Predicts `steps` days into the future by feeding each prediction
        back as input for the next step.
        
        Args:
            last_sequence: Array of last `seq_length` price values.
            steps: Number of future steps to predict.
        
        Returns:
            np.ndarray of shape (steps,) with predicted prices.
        """
        if torch is None or self.model is None:
            raise RuntimeError("Model or PyTorch is not available.")
        
        self.model.eval()
        predictions = []
        
        # Scale the initial sequence
        current_seq = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        with torch.no_grad():
            for _ in range(steps):
                input_tensor = torch.FloatTensor(current_seq[-self.seq_length:]).reshape(1, self.seq_length, 1)
                pred_scaled = self.model(input_tensor).item()
                predictions.append(pred_scaled)
                # Append prediction to the sequence for the next step
                current_seq = np.append(current_seq, pred_scaled)
        
        # Inverse transform all predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        return predictions

    def predict_with_uncertainty(self, last_sequence, steps=30, n_samples=50):
        """Multi-step forecast with MC Dropout uncertainty estimation.
        
        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.
        
        Args:
            last_sequence: Array of last `seq_length` price values.
            steps: Number of future steps.
            n_samples: Number of MC Dropout samples.
        
        Returns:
            dict with keys: mean, lower, upper, std (each np.ndarray of shape (steps,))
        """
        if torch is None or self.model is None:
            raise RuntimeError("Model or PyTorch is not available.")
        
        # Enable dropout during inference (MC Dropout)
        self.model.train()  # Keep dropout active
        
        all_preds = []
        for _ in range(n_samples):
            current_seq = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
            sample_preds = []
            
            with torch.no_grad():
                for __ in range(steps):
                    input_tensor = torch.FloatTensor(current_seq[-self.seq_length:]).reshape(1, self.seq_length, 1)
                    pred_scaled = self.model(input_tensor).item()
                    sample_preds.append(pred_scaled)
                    current_seq = np.append(current_seq, pred_scaled)
            
            sample_preds = np.array(sample_preds).reshape(-1, 1)
            sample_preds = self.scaler.inverse_transform(sample_preds).flatten()
            all_preds.append(sample_preds)
        
        self.model.eval()
        
        all_preds = np.array(all_preds)  # shape: (n_samples, steps)
        return {
            'mean': np.mean(all_preds, axis=0),
            'lower': np.percentile(all_preds, 5, axis=0),   # 90% CI lower
            'upper': np.percentile(all_preds, 95, axis=0),   # 90% CI upper
            'std': np.std(all_preds, axis=0),
        }


if __name__ == "__main__":
    from models.evaluation import calculate_metrics, print_evaluation_report
    
    # Test on a single series
    data_file = "/Users/fahmiprasanda/Documents/python/lstm_prophet/food_prices_real.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        forecaster = LSTMForecaster()
        
        # 1. Siapkan dataset
        X, y = forecaster.prepare_data(df, 'DKI Jakarta', 'Beras')
        
        # 2. Pemisahan Data (Skenario Pengujian) - 80% Train, 20% Test
        X_train, X_test, y_train, y_test = forecaster.split_data(X, y, test_size=0.2)
        print(f"Dataset split: {len(X_train)} Train sequences, {len(X_test)} Test sequences.")
        
        # 3. Training
        forecaster.train_single_series(X_train, y_train, epochs=10)
        
        # 4. Evaluasi (Metrik Evaluasi)
        forecaster.model.eval()
        with torch.no_grad():
            y_pred_scaled = forecaster.model(X_test)
            
            # Kembalikan ke skala harga asli dari -1:1
            y_pred = forecaster.scaler.inverse_transform(y_pred_scaled.numpy().reshape(-1, 1))
            y_true = forecaster.scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
            
            metrics = calculate_metrics(y_true, y_pred, model_name="LSTM (Bidirectional)")
            print_evaluation_report(metrics)
            
        # 5. Multi-step prediction
        last_30 = df[(df['province'] == 'DKI Jakarta') & (df['commodity'] == 'Beras')]['price'].values[-30:]
        multi_preds = forecaster.predict_multi_step(last_30, steps=7)
        print(f"\n7-day multi-step forecast:")
        for i, p in enumerate(multi_preds):
            print(f"  Day {i+1}: IDR {p:,.0f}")
        
        # 6. Uncertainty estimation
        uncertainty = forecaster.predict_with_uncertainty(last_30, steps=7, n_samples=20)
        print(f"\n7-day forecast with uncertainty (90% CI):")
        for i in range(7):
            print(f"  Day {i+1}: {uncertainty['mean'][i]:,.0f} [{uncertainty['lower'][i]:,.0f} - {uncertainty['upper'][i]:,.0f}]")
    else:
        print("Data file not found.")
