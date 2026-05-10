"""
Smart Ensemble model for Agri-AI EWS.
Combines Prophet, LSTM, and TFT predictions with adaptive weighting
based on recent model performance.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SmartEnsemble:
    """Adaptive ensemble that weights models based on recent accuracy.
    
    Strategy:
    - If all 3 models available: TFT(0.4) + Prophet(0.35) + LSTM(0.25)
    - Weights adjust based on recent MAPE on validation data
    - Fallback cascade: TFT → LSTM → Prophet
    - Confidence interval: union of individual model intervals
    """

    def __init__(self, default_weights=None):
        """
        Args:
            default_weights: dict like {'prophet': 0.35, 'lstm': 0.25, 'tft': 0.4}
        """
        self.weights = default_weights or {
            'prophet': 0.40,
            'lstm': 0.30,
            'tft': 0.30,
        }
        self.performance_history = {}

    def combine_forecasts(self, predictions: dict, target_days: int = None) -> dict:
        """Combine predictions from multiple models.
        
        Args:
            predictions: dict of model_name → dict with keys:
                - 'mean': np.ndarray or float (predicted values)
                - 'lower': np.ndarray or float (lower bound, optional)
                - 'upper': np.ndarray or float (upper bound, optional)
            target_days: Specific day index to extract (for single-point forecast).
        
        Returns:
            dict with 'mean', 'lower', 'upper', 'model_weights', 'models_used'
        """
        if not predictions:
            return {'mean': None, 'lower': None, 'upper': None, 
                    'model_weights': {}, 'models_used': []}

        # Filter available models
        available = {k: v for k, v in predictions.items() if v is not None and v.get('mean') is not None}
        
        if not available:
            return {'mean': None, 'lower': None, 'upper': None, 
                    'model_weights': {}, 'models_used': []}

        # Normalize weights for available models only
        total_weight = sum(self.weights.get(m, 0.2) for m in available)
        norm_weights = {m: self.weights.get(m, 0.2) / total_weight for m in available}

        # Compute weighted ensemble
        means = []
        lowers = []
        uppers = []

        for model_name, pred in available.items():
            w = norm_weights[model_name]
            mean_val = np.atleast_1d(pred['mean'])
            means.append(w * mean_val)

            if pred.get('lower') is not None:
                lowers.append(np.atleast_1d(pred['lower']))
            if pred.get('upper') is not None:
                uppers.append(np.atleast_1d(pred['upper']))

        # Weighted mean
        # Align lengths (use shortest)
        min_len = min(len(m) for m in means)
        ensemble_mean = np.sum([m[:min_len] for m in means], axis=0)

        # Confidence interval: take widest bounds across models
        if lowers:
            ensemble_lower = np.min([l[:min_len] for l in lowers], axis=0)
        else:
            ensemble_lower = ensemble_mean * 0.90

        if uppers:
            ensemble_upper = np.max([u[:min_len] for u in uppers], axis=0)
        else:
            ensemble_upper = ensemble_mean * 1.10

        # Extract single point if requested
        if target_days is not None and target_days < min_len:
            ensemble_mean = ensemble_mean[target_days]
            ensemble_lower = ensemble_lower[target_days]
            ensemble_upper = ensemble_upper[target_days]

        return {
            'mean': ensemble_mean,
            'lower': ensemble_lower,
            'upper': ensemble_upper,
            'model_weights': norm_weights,
            'models_used': list(available.keys()),
        }

    def update_weights_from_errors(self, model_errors: dict):
        """Update model weights based on recent prediction errors.
        
        Args:
            model_errors: dict of model_name → MAPE (lower is better)
        """
        if not model_errors:
            return

        # Inverse MAPE weighting (lower MAPE = higher weight)
        inverse = {m: 1.0 / max(e, 0.01) for m, e in model_errors.items()}
        total = sum(inverse.values())
        
        new_weights = {m: v / total for m, v in inverse.items()}
        
        # Smooth update (70% new, 30% old) to avoid drastic changes
        for m in new_weights:
            old_w = self.weights.get(m, 0.2)
            self.weights[m] = 0.7 * new_weights[m] + 0.3 * old_w

        # Record history
        self.performance_history[pd.Timestamp.now().isoformat()] = {
            'errors': model_errors.copy(),
            'weights': self.weights.copy(),
        }

        logger.info(f"Updated ensemble weights: {self.weights}")

    def get_forecast_with_distance_weighting(self, predictions: dict, days_ahead: int) -> dict:
        """Adjust weights based on forecast horizon distance.
        
        Short-term (1-7 days): favor LSTM
        Medium-term (8-30 days): balanced
        Long-term (31+ days): favor Prophet/TFT
        
        Args:
            predictions: Same format as combine_forecasts.
            days_ahead: Number of days into the future.
        
        Returns:
            Same format as combine_forecasts.
        """
        # Save original weights
        original_weights = self.weights.copy()

        # Adjust for horizon
        if days_ahead <= 7:
            # Short-term: LSTM excels
            self.weights = {
                'prophet': 0.25,
                'lstm': 0.50,
                'tft': 0.25,
            }
        elif days_ahead <= 30:
            # Medium-term: balanced
            self.weights = {
                'prophet': 0.35,
                'lstm': 0.30,
                'tft': 0.35,
            }
        else:
            # Long-term: Prophet/TFT for trend
            self.weights = {
                'prophet': 0.45,
                'lstm': 0.10,
                'tft': 0.45,
            }

        result = self.combine_forecasts(predictions, target_days=days_ahead - 1 if days_ahead > 0 else 0)
        
        # Restore original weights
        self.weights = original_weights
        
        return result
