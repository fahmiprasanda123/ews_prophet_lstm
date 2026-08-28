"""
Conventional Forecasting Methods for Agri-AI EWS.
Provides baseline models (Naïve Seasonal, Moving Average, ARIMA/SARIMA)
for benchmarking against AI models (Prophet, BiLSTM, TFT).
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ConventionalForecaster:
    """Conventional time-series forecasting methods as baselines.
    
    Methods:
        - Naïve Seasonal: repeats prices from the same period last year
        - Moving Average (SMA-30): simple 30-day moving average
        - ARIMA/SARIMA: classical statistical time-series model
    """

    def __init__(self, df):
        """
        Args:
            df: DataFrame with columns: date, province, commodity, price.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def _get_series(self, province, commodity):
        """Extract and sort the price series for a given province/commodity."""
        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date').reset_index(drop=True)
        return series

    def _split_series(self, series, test_size=0.2):
        """Chronological train/test split."""
        split_idx = int(len(series) * (1 - test_size))
        return series.iloc[:split_idx], series.iloc[split_idx:]

    # -----------------------------------------------------------------
    # 1. Naïve Seasonal Forecast
    # -----------------------------------------------------------------
    def naive_seasonal_forecast(self, province, commodity, test_size=0.2,
                                 seasonal_period=365):
        """Seasonal Naïve: predict using the value from `seasonal_period` days ago.
        
        For daily data, seasonal_period=365 means "same day last year".
        Falls back to last known value if insufficient history.
        
        Args:
            province: Province name.
            commodity: Commodity name.
            test_size: Fraction for test set.
            seasonal_period: Lag period in days (default 365).
        
        Returns:
            dict with keys: y_true, y_pred, train_size, test_size
        """
        series = self._get_series(province, commodity)
        train, test = self._split_series(series, test_size)

        train_prices = train['price'].values
        predictions = []

        for i in range(len(test)):
            # Index in the full series
            full_idx = len(train) + i
            lag_idx = full_idx - seasonal_period

            if lag_idx >= 0:
                predictions.append(series['price'].iloc[lag_idx])
            else:
                # Fallback: use last training value
                predictions.append(train_prices[-1])

        return {
            'y_true': test['price'].values,
            'y_pred': np.array(predictions),
            'train_size': len(train),
            'test_size': len(test),
            'dates': test['date'].values,
        }

    # -----------------------------------------------------------------
    # 2. Moving Average (SMA) Forecast
    # -----------------------------------------------------------------
    def moving_average_forecast(self, province, commodity, test_size=0.2,
                                 window=30):
        """Simple Moving Average: predict using the mean of last `window` days.
        
        At each test point, uses a rolling window of the most recent `window`
        observed values (walk-forward, no future leakage).
        
        Args:
            province: Province name.
            commodity: Commodity name.
            test_size: Fraction for test set.
            window: Rolling window size in days (default 30).
        
        Returns:
            dict with keys: y_true, y_pred, train_size, test_size
        """
        series = self._get_series(province, commodity)
        train, test = self._split_series(series, test_size)

        all_prices = series['price'].values
        train_end = len(train)
        predictions = []

        for i in range(len(test)):
            # Use prices up to (but not including) the current test point
            available = all_prices[:train_end + i]
            # Take the last `window` values
            window_data = available[-window:]
            predictions.append(np.mean(window_data))

        return {
            'y_true': test['price'].values,
            'y_pred': np.array(predictions),
            'train_size': len(train),
            'test_size': len(test),
            'dates': test['date'].values,
        }

    # -----------------------------------------------------------------
    # 3. ARIMA / SARIMA Forecast
    # -----------------------------------------------------------------
    def arima_forecast(self, province, commodity, test_size=0.2,
                       order=(5, 1, 0), seasonal_order=None):
        """ARIMA/SARIMA forecast using statsmodels.
        
        Trains on training set and produces one-step-ahead forecasts
        for the test set (walk-forward without re-fitting for speed).
        
        Args:
            province: Province name.
            commodity: Commodity name.
            test_size: Fraction for test set.
            order: ARIMA (p, d, q) order. Default (5,1,0).
            seasonal_order: Optional SARIMA seasonal order (P,D,Q,s).
                           Set to (1,1,0,7) for weekly seasonality.
        
        Returns:
            dict with keys: y_true, y_pred, train_size, test_size
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.error("statsmodels not installed. Run: pip install statsmodels")
            return None

        series = self._get_series(province, commodity)
        train, test = self._split_series(series, test_size)

        train_prices = train['price'].values
        test_prices = test['price'].values

        try:
            # Fit model on training data
            if seasonal_order is not None:
                model = SARIMAX(
                    train_prices,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(
                    train_prices,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(method_kwargs={'maxiter': 200})

            # Forecast the test period
            forecast = fitted.forecast(steps=len(test))
            predictions = np.array(forecast)

            # Clip negative predictions to zero (prices can't be negative)
            predictions = np.clip(predictions, 0, None)

            return {
                'y_true': test_prices,
                'y_pred': predictions,
                'train_size': len(train),
                'test_size': len(test),
                'dates': test['date'].values,
            }

        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            return None

    # -----------------------------------------------------------------
    # Multi-step forecast for backtesting
    # -----------------------------------------------------------------
    def naive_multi_step(self, train_prices, steps, seasonal_period=365):
        """Multi-step Naïve Seasonal forecast for backtesting folds."""
        preds = []
        for i in range(steps):
            idx = len(train_prices) - seasonal_period + i
            if 0 <= idx < len(train_prices):
                preds.append(train_prices[idx])
            else:
                preds.append(train_prices[-1])
        return np.array(preds)

    def sma_multi_step(self, train_prices, steps, window=30):
        """Multi-step SMA forecast for backtesting folds.
        
        Uses a recursive approach: each predicted value is appended
        to the series for the next prediction.
        """
        extended = list(train_prices)
        preds = []
        for _ in range(steps):
            pred = np.mean(extended[-window:])
            preds.append(pred)
            extended.append(pred)
        return np.array(preds)

    def arima_multi_step(self, train_prices, steps, order=(5, 1, 0)):
        """Multi-step ARIMA forecast for backtesting folds."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train_prices, order=order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fitted = model.fit(method_kwargs={'maxiter': 200})
                forecast = fitted.forecast(steps=steps)
                return np.clip(np.array(forecast), 0, None)
        except Exception as e:
            logger.warning(f"ARIMA multi-step failed: {e}")
            return None
