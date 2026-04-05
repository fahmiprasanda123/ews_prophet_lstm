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
    # Test on a single series
    data_file = "/Users/fahmisp/Documents/python/lstm_prophet/food_prices_indonesia.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        forecaster = FoodPriceProphet(df)
        forecast = forecaster.train_and_forecast('DKI Jakarta', 'Beras', periods=30)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    else:
        print(f"Data file not found at {data_file}. Run data_generator.py first.")
