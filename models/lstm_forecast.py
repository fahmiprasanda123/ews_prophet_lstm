import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

class LSTMForecaster:
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.model = PriceLSTM()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def prepare_data(self, df, province, commodity):
        subset = df[(df['province'] == province) & (df['commodity'] == commodity)]['price'].values
        subset = subset.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(subset)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i:i+self.seq_length])
            y.append(scaled_data[i+self.seq_length])
            
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def train_single_series(self, X, y, epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
        self.model.eval()
        with torch.no_grad():
            scaled_seq = self.scaler.transform(last_sequence.reshape(-1, 1))
            input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0)
            prediction = self.model(input_tensor)
            return self.scaler.inverse_transform(prediction.numpy())

if __name__ == "__main__":
    # Test on a single series
    data_file = "/Users/fahmiprasanda/Documents/python/lstm_prophet/food_prices_indonesia.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        forecaster = LSTMForecaster()
        X, y = forecaster.prepare_data(df, 'DKI Jakarta', 'Beras')
        forecaster.train_single_series(X, y, epochs=10)
        
        # Predict next value
        last_30 = df[(df['province'] == 'DKI Jakarta') & (df['commodity'] == 'Beras')]['price'].values[-30:]
        pred = forecaster.predict(last_30)
        print(f"Next day prediction: {pred[0][0]}")
    else:
        print("Data file not found.")
