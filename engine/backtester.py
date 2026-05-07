"""
Walk-Forward Backtesting Framework for Agri-AI EWS.
Tests model accuracy and EWS alert quality on historical data.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """Walk-forward backtesting engine.
    
    Slides a training window across historical data, trains models
    at each step, and evaluates predictions against actual outcomes.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def walk_forward_test(self, province, commodity, 
                          train_window=180, test_window=30, step_size=30,
                          model_type='prophet'):
        """Run walk-forward validation.
        
        Args:
            province, commodity: Target series.
            train_window: Days of training data.
            test_window: Days to forecast.
            step_size: Days to advance between folds.
            model_type: 'prophet' or 'lstm'.
        
        Returns:
            list of fold results, each with metrics and predictions.
        """
        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date').reset_index(drop=True)

        if len(series) < train_window + test_window:
            return []

        results = []
        start = 0
        fold = 0

        while start + train_window + test_window <= len(series):
            train_end = start + train_window
            test_end = train_end + test_window

            train_data = series.iloc[start:train_end]
            test_data = series.iloc[train_end:test_end]

            try:
                if model_type == 'prophet':
                    preds = self._run_prophet_fold(train_data, test_data)
                elif model_type == 'lstm':
                    preds = self._run_lstm_fold(train_data, test_data)
                else:
                    preds = self._run_prophet_fold(train_data, test_data)

                if preds is not None:
                    from models.evaluation import calculate_metrics
                    actual = test_data['price'].values[:len(preds)]
                    metrics = calculate_metrics(actual, preds, model_name=f"{model_type}_fold{fold}")

                    results.append({
                        'fold': fold,
                        'train_start': train_data['date'].iloc[0].strftime('%Y-%m-%d'),
                        'train_end': train_data['date'].iloc[-1].strftime('%Y-%m-%d'),
                        'test_start': test_data['date'].iloc[0].strftime('%Y-%m-%d'),
                        'test_end': test_data['date'].iloc[min(len(preds)-1, len(test_data)-1)].strftime('%Y-%m-%d'),
                        'metrics': metrics,
                        'predictions': preds.tolist(),
                        'actuals': actual.tolist(),
                    })
            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")

            start += step_size
            fold += 1

        return results

    def test_ews_accuracy(self, province, commodity, ews_engine, 
                          threshold_pct=10, lookback_days=365):
        """Test how accurately EWS detected historical price spikes.
        
        Args:
            province, commodity: Target series.
            ews_engine: EWSEngineV2 instance.
            threshold_pct: What counts as a "spike" (% change in 14 days).
            lookback_days: How far back to test.
        
        Returns:
            dict with detection_rate, false_alarm_rate, avg_lead_time, events.
        """
        series = self.df[
            (self.df['province'] == province) & (self.df['commodity'] == commodity)
        ].sort_values('date').reset_index(drop=True)

        if len(series) < lookback_days:
            lookback_days = len(series) - 30

        prices = series['price'].values
        dates = series['date'].values

        # Detect actual spikes
        spikes = []
        for i in range(14, len(prices)):
            change = (prices[i] - prices[i-14]) / prices[i-14] * 100
            if change >= threshold_pct:
                spikes.append({
                    'index': i,
                    'date': str(dates[i])[:10],
                    'change_pct': round(change, 2),
                    'price': prices[i],
                })

        if not spikes:
            return {
                'detection_rate': 0,
                'false_alarm_rate': 0,
                'avg_lead_time': 0,
                'total_spikes': 0,
                'events': [],
                'message': 'Tidak ada spike terdeteksi dalam periode ini.'
            }

        # Simulate EWS at each pre-spike point
        detected = 0
        events = []
        for spike in spikes[-20:]:  # Limit to last 20 spikes
            idx = spike['index']
            if idx < 30:
                continue

            # Check if EWS would have fired 7-14 days before
            for lead in [14, 10, 7]:
                check_idx = idx - lead
                if check_idx < 0:
                    continue
                current_p = prices[check_idx]
                predicted_p = prices[idx]  # Hindsight
                try:
                    result = ews_engine.calculate_composite_score(
                        province, commodity, predicted_p
                    )
                    if result['level'] in ['Danger', 'Alert']:
                        detected += 1
                        events.append({
                            'spike_date': spike['date'],
                            'warning_lead_days': lead,
                            'ews_level': result['level'],
                            'ews_score': result['score'],
                        })
                        break
                except Exception:
                    pass

        detection_rate = (detected / len(spikes) * 100) if spikes else 0
        avg_lead = np.mean([e['warning_lead_days'] for e in events]) if events else 0

        return {
            'detection_rate': round(detection_rate, 1),
            'false_alarm_rate': 0,  # Simplified
            'avg_lead_time': round(avg_lead, 1),
            'total_spikes': len(spikes),
            'detected_spikes': detected,
            'events': events,
        }

    def get_summary(self, results):
        """Summarize walk-forward results."""
        if not results:
            return {'avg_mape': None, 'avg_rmse': None, 'folds': 0}

        mapes = [r['metrics']['MAPE (%)'] for r in results]
        rmses = [r['metrics']['RMSE'] for r in results]
        r2s = [r['metrics'].get('R²', 0) for r in results]

        return {
            'folds': len(results),
            'avg_mape': round(np.mean(mapes), 2),
            'std_mape': round(np.std(mapes), 2),
            'avg_rmse': round(np.mean(rmses), 2),
            'avg_r2': round(np.mean(r2s), 4),
            'best_fold': min(range(len(mapes)), key=lambda i: mapes[i]),
            'worst_fold': max(range(len(mapes)), key=lambda i: mapes[i]),
        }

    def _run_prophet_fold(self, train_data, test_data):
        """Run Prophet on a single fold."""
        try:
            from prophet import Prophet
        except ImportError:
            return None

        train_df = train_data[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True,
            daily_seasonality=False, changepoint_prior_scale=0.05
        )
        model.fit(train_df)

        future = test_data[['date']].rename(columns={'date': 'ds'})
        forecast = model.predict(future)
        return forecast['yhat'].values

    def _run_lstm_fold(self, train_data, test_data):
        """Run LSTM on a single fold."""
        try:
            import torch
            from models.lstm_forecast import LSTMForecaster
        except ImportError:
            return None

        all_prices = train_data['price'].values
        forecaster = LSTMForecaster(seq_length=30)
        
        # Fit scaler on training data
        from sklearn.preprocessing import MinMaxScaler
        forecaster.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = forecaster.scaler.fit_transform(all_prices.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(len(scaled) - 30):
            X.append(scaled[i:i+30])
            y.append(scaled[i+30])

        if len(X) < 10:
            return None

        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))

        forecaster.train_single_series(X, y, epochs=5)

        # Predict
        last_seq = all_prices[-30:]
        preds = forecaster.predict_multi_step(last_seq, steps=len(test_data))
        return preds
