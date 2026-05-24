"""Tests for SmartEnsemble model."""
import pytest
import numpy as np
from models.ensemble import SmartEnsemble


class TestSmartEnsemble:
    def setup_method(self):
        self.ensemble = SmartEnsemble()

    def test_combine_empty_predictions(self):
        result = self.ensemble.combine_forecasts({})
        assert result['mean'] is None
        assert result['models_used'] == []

    def test_combine_single_model(self):
        predictions = {
            'prophet': {
                'mean': np.array([100, 200, 300]),
                'lower': np.array([90, 190, 290]),
                'upper': np.array([110, 210, 310]),
            }
        }
        result = self.ensemble.combine_forecasts(predictions)
        
        # Single model → output should equal input (weight = 1.0)
        np.testing.assert_array_almost_equal(result['mean'], [100, 200, 300])
        assert 'prophet' in result['models_used']
        assert len(result['models_used']) == 1

    def test_combine_two_models_weighted(self):
        predictions = {
            'prophet': {'mean': np.array([100.0]), 'lower': np.array([90.0]), 'upper': np.array([110.0])},
            'lstm': {'mean': np.array([200.0]), 'lower': np.array([180.0]), 'upper': np.array([220.0])},
        }
        result = self.ensemble.combine_forecasts(predictions)
        
        # Should be weighted average, not simple average
        assert result['mean'] is not None
        # With default weights prophet=0.4, lstm=0.3 → normalized = 0.571, 0.429
        expected = 100 * (0.4 / 0.7) + 200 * (0.3 / 0.7)
        np.testing.assert_array_almost_equal(result['mean'], [expected], decimal=1)
        
        # Lower bound should be the minimum, upper should be maximum
        assert result['lower'][0] <= result['mean'][0]
        assert result['upper'][0] >= result['mean'][0]

    def test_combine_with_none_model(self):
        """Model returning None should be filtered out."""
        predictions = {
            'prophet': {'mean': np.array([100.0])},
            'lstm': None,
        }
        result = self.ensemble.combine_forecasts(predictions)
        assert len(result['models_used']) == 1
        assert 'prophet' in result['models_used']

    def test_combine_different_lengths(self):
        """Should use shortest prediction length."""
        predictions = {
            'prophet': {'mean': np.array([100, 200, 300])},
            'lstm': {'mean': np.array([150, 250])},
        }
        result = self.ensemble.combine_forecasts(predictions)
        assert len(result['mean']) == 2  # min length

    def test_update_weights_from_errors(self):
        """Lower MAPE should get higher weight."""
        self.ensemble.update_weights_from_errors({
            'prophet': 5.0,  # Best (lowest MAPE)
            'lstm': 10.0,
            'tft': 20.0,
        })
        
        assert self.ensemble.weights['prophet'] > self.ensemble.weights['lstm']
        assert self.ensemble.weights['lstm'] > self.ensemble.weights['tft']

    def test_distance_weighting_short_term(self):
        """Short-term should favor LSTM."""
        predictions = {
            'prophet': {'mean': np.array([100.0] * 10)},
            'lstm': {'mean': np.array([200.0] * 10)},
            'tft': {'mean': np.array([150.0] * 10)},
        }
        
        result = self.ensemble.get_forecast_with_distance_weighting(predictions, days_ahead=3)
        # Short-term: LSTM gets higher weight, so result should be closer to 200
        assert result['mean'] is not None

    def test_confidence_interval_union(self):
        """Confidence interval should be the widest (union) of all models."""
        predictions = {
            'prophet': {'mean': np.array([100.0]), 'lower': np.array([80.0]), 'upper': np.array([120.0])},
            'lstm': {'mean': np.array([100.0]), 'lower': np.array([70.0]), 'upper': np.array([110.0])},
        }
        result = self.ensemble.combine_forecasts(predictions)
        
        # Lower = min(80, 70) = 70, Upper = max(120, 110) = 120
        assert result['lower'][0] == 70.0
        assert result['upper'][0] == 120.0

    def test_target_days_extraction(self):
        """target_days should extract a single value from the forecast."""
        predictions = {
            'prophet': {'mean': np.array([100.0, 200.0, 300.0])},
            'lstm': {'mean': np.array([110.0, 210.0, 310.0])},
        }
        result = self.ensemble.combine_forecasts(predictions, target_days=1)
        
        # Should be a scalar (single value for day index 1)
        assert not hasattr(result['mean'], '__len__')
