import pandas as pd
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
import os

class FoodPriceProphet:
    def __init__(self, df):
        self.df = df
        self.models = {}
        
    def prepare_data(self, province, commodity):
        # Prophet expects 'ds' and 'y' columns
        subset = self.df[(self.df['province'] == province) & (self.df['commodity'] == commodity)]
        prophet_df = subset[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        return prophet_df
        
    def split_data(self, df, test_size=0.2):
        """
        Membagi data Prophet (berisi ds, y) menjadi Train dan Test berdasarkan kronologi waktu.
        """
        df = df.sort_values(by='ds').reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df
    
    def train_and_forecast(self, province, commodity, periods=30):
        if Prophet is None:
            raise RuntimeError("Prophet library is not installed correctly. Forecast cannot be performed.")
            
        print(f"Training Prophet for {commodity} in {province}...")
        p_df = self.prepare_data(province, commodity)
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(p_df)
        
        # Forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Store model
        model_key = f"{province}_{commodity}"
        self.models[model_key] = model
        
        return forecast

if __name__ == "__main__":
    from models.evaluation import calculate_metrics, print_evaluation_report
    
    # Test on a single series
    data_file = "/Users/fahmiprasanda/Documents/python/lstm_prophet/food_prices_real.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        forecaster = FoodPriceProphet(df)
        
        # 1. Siapkan Dataset
        p_df = forecaster.prepare_data('DKI Jakarta', 'Beras')
        
        # 2. Pemisahan Data (Skenario Pengujian) - 20% Testing
        train_df, test_df = forecaster.split_data(p_df, test_size=0.2)
        print(f"Dataset split: {len(train_df)} Train days, {len(test_df)} Test days.")
        
        # 3. Training & Evaluation Simulation
        if Prophet is not None:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(train_df)
            
            # 4. Prediksi pada area tanggal Test Set
            future = test_df[['ds']]
            forecast = model.predict(future)
            
            # Evaluasi
            y_pred = forecast['yhat'].values
            y_true = test_df['y'].values
            metrics = calculate_metrics(y_true, y_pred, model_name="Prophet")
            print_evaluation_report(metrics)
            
            # 5. Prediksi Masa Depan (Deployment scenario)
            print("\nNext future prediction (30 periods):")
            future_real = model.make_future_dataframe(periods=30)
            forecast_real = model.predict(future_real)
            print(forecast_real[['ds', 'yhat']].tail())
    else:
        print(f"Data file not found at {data_file}. Run data_generator.py first.")
