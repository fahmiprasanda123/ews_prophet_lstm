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
    
    def train_and_forecast(self, province, commodity, periods=30, weather_df=None):
        """Train Prophet model and generate forecast with confidence intervals.
        
        Args:
            province: Province name.
            commodity: Commodity name.
            periods: Number of future periods to forecast.
            weather_df: Optional DataFrame with weather regressors
                        (columns: rainfall_mm, enso_index, is_wet_season).
        
        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, trend, 
            yearly, weekly (and regressor components if weather is provided).
        """
        if Prophet is None:
            raise RuntimeError("Prophet library is not installed correctly. Forecast cannot be performed.")
            
        print(f"Training Prophet for {commodity} in {province}...")
        p_df = self.prepare_data(province, commodity)
        
        # Initialize model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.90  # 90% confidence interval
        )
        
        # Add external regressors if weather data is available
        has_weather = False
        if weather_df is not None and not weather_df.empty:
            try:
                # Merge weather data with prophet data
                p_df = p_df.copy()
                p_df['ds'] = pd.to_datetime(p_df['ds'])
                weather_df = weather_df.copy()
                weather_df.index = pd.to_datetime(weather_df.index)
                
                # Align weather data with price data
                for col in ['rainfall_mm', 'enso_index', 'is_wet_season']:
                    if col in weather_df.columns:
                        p_df[col] = p_df['ds'].map(
                            weather_df[col].to_dict()
                        ).fillna(weather_df[col].mean() if len(weather_df) > 0 else 0)
                        model.add_regressor(col)
                        has_weather = True
            except Exception as e:
                print(f"Warning: Could not add weather regressors: {e}")
                has_weather = False
        
        model.fit(p_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressor values for future dates
        if has_weather:
            for col in ['rainfall_mm', 'enso_index', 'is_wet_season']:
                if col in p_df.columns:
                    # For future dates, use seasonal average as proxy
                    existing_vals = p_df.set_index('ds')[col]
                    future[col] = future['ds'].map(existing_vals.to_dict())
                    # Fill future dates with monthly average
                    if col == 'rainfall_mm':
                        monthly_avg = p_df.groupby(p_df['ds'].dt.month)[col].mean()
                        future[col] = future.apply(
                            lambda row: monthly_avg.get(row['ds'].month, 0) 
                            if pd.isna(row[col]) else row[col],
                            axis=1
                        )
                    elif col == 'is_wet_season':
                        future[col] = future['ds'].apply(
                            lambda d: 1 if d.month in [11, 12, 1, 2, 3, 4] else 0
                        ).where(future[col].isna(), future[col])
                    else:
                        future[col] = future[col].ffill().bfill().fillna(0)
        
        forecast = model.predict(future)
        
        # Store model
        model_key = f"{province}_{commodity}"
        self.models[model_key] = model
        
        return forecast

    def get_components(self, province, commodity):
        """Get decomposition components for an already-trained model."""
        model_key = f"{province}_{commodity}"
        model = self.models.get(model_key)
        if model is None:
            return None
        return model

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
            print(forecast_real[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    else:
        print(f"Data file not found at {data_file}. Run data_generator.py first.")
