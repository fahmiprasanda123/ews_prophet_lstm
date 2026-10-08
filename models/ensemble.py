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

    # ------------------------------------------------------------------ #
    # Pembobotan berbasis validasi (dipakai protokol evaluasi tesis)
    # ------------------------------------------------------------------ #
    def fit_weights_grid(self, val_predictions: dict, val_actual, step: float = 0.05,
                         metric: str = 'MAPE (%)') -> dict:
        """Cari bobot terbaik dengan pencarian grid pada periode validasi.

        Semua kombinasi bobot non-negatif berkelipatan `step` yang berjumlah 1
        dicoba; kombinasi dengan nilai `metric` terkecil dipilih. Bobot hasil
        pencarian disimpan di self.weights sehingga combine_forecasts() dan
        combine_series() langsung memakainya.

        Args:
            val_predictions: dict nama_model -> pd.Series prediksi (indeks tanggal).
            val_actual: pd.Series harga aktual (indeks tanggal).
            step: Resolusi grid (0,05 = kelipatan 5%).
            metric: Metrik galat yang diminimalkan (kunci dari calculate_metrics).

        Returns:
            dict: weights, best_score, validation_scores, best_single, n_days.
        """
        from itertools import product
        from models.evaluation import calculate_metrics

        names = [m for m, s in val_predictions.items() if s is not None and len(s) > 0]
        if not names:
            raise ValueError("Tidak ada prediksi validasi untuk mencari bobot.")
        frame = pd.concat({m: val_predictions[m] for m in names}, axis=1).dropna()
        actual = val_actual.reindex(frame.index)
        keep = actual.notna()
        frame, actual = frame[keep], actual[keep]

        scores = {m: calculate_metrics(actual.values, frame[m].values, m)[metric] for m in names}
        n_steps = int(round(1 / step))
        best_score, best_w = np.inf, None
        for combo in product(range(n_steps + 1), repeat=len(names)):
            if sum(combo) != n_steps:
                continue
            w = np.array(combo, dtype=float) / n_steps
            score = calculate_metrics(actual.values, frame.values @ w)[metric]
            if score < best_score - 1e-12:
                best_score, best_w = score, w

        self.weights = {m: float(w) for m, w in zip(names, best_w)}
        self.validation_scores = scores
        self.best_single = min(scores, key=scores.get)
        return {
            'weights': dict(self.weights),
            'best_score': float(best_score),
            'validation_scores': {m: float(v) for m, v in scores.items()},
            'best_single': self.best_single,
            'n_days': int(len(frame)),
            'metric': metric,
            'step': step,
        }

    def combine_series(self, predictions: dict, fallback_cv: float = 0.15):
        """Gabungkan prediksi beberapa model per TANGGAL memakai self.weights.

        Fallback: bila koefisien variasi antarprediksi model pada suatu tanggal
        melebihi `fallback_cv`, ensemble memakai prediksi model dengan skor
        validasi terbaik (self.best_single) untuk tanggal tersebut.

        Args:
            predictions: dict nama_model -> pd.Series (indeks tanggal).
            fallback_cv: Ambang koefisien variasi; None untuk mematikan fallback.

        Returns:
            (pd.Series ensemble, pd.Series bool penanda fallback)
        """
        names = [m for m in self.weights if m in predictions and predictions[m] is not None]
        if not names:
            raise ValueError("Tidak ada model yang cocok dengan bobot ensemble.")
        frame = pd.concat({m: predictions[m] for m in names}, axis=1).dropna()
        w = np.array([self.weights[m] for m in names], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
        ensemble = pd.Series(frame.values @ w, index=frame.index, name='Smart Ensemble')
        flag = pd.Series(False, index=frame.index, name='fallback')
        best = getattr(self, 'best_single', None)
        if fallback_cv is not None and len(names) > 1 and best in frame.columns:
            cv = frame.std(axis=1, ddof=0) / frame.mean(axis=1).abs()
            flag = (cv > fallback_cv).rename('fallback')
            ensemble[flag] = frame.loc[flag, best]
        return ensemble, flag

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
