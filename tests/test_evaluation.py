"""Tests for evaluation metrics module."""
import pytest
import numpy as np
from models.evaluation import (
    calculate_mape, calculate_smape, calculate_directional_accuracy,
    calculate_metrics, compare_models, print_evaluation_report
)


class TestCalculateMAPE:
    def test_basic_mape(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        mape = calculate_mape(y_true, y_pred)
        # |10/100| + |10/200| + |10/300| = 0.1 + 0.05 + 0.0333 → mean = 0.0611
        assert 5 < mape < 8  # ~6.11%

    def test_perfect_prediction(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        assert calculate_mape(y_true, y_pred) == 0.0

    def test_zero_values_handled(self):
        """MAPE should skip zero values in y_true to avoid division by zero."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 210])
        mape = calculate_mape(y_true, y_pred)
        # Only non-zero: |10/100| + |10/200| = 0.1 + 0.05 = 0.075
        assert 5 < mape < 10

    def test_all_zeros(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        assert calculate_mape(y_true, y_pred) == 0.0


class TestCalculateSMAPE:
    def test_basic_smape(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        smape = calculate_smape(y_true, y_pred)
        assert 0 < smape < 10

    def test_perfect_prediction(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([100.0, 200.0])
        assert calculate_smape(y_true, y_pred) == 0.0

    def test_symmetric(self):
        """SMAPE should be symmetric — same result if pred and true are swapped."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 220.0, 280.0])
        smape1 = calculate_smape(y_true, y_pred)
        smape2 = calculate_smape(y_pred, y_true)
        np.testing.assert_almost_equal(smape1, smape2)

    def test_both_zeros(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([0.0, 0.0])
        assert calculate_smape(y_true, y_pred) == 0.0


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        y_true = np.array([100, 110, 120, 130])  # Always up
        y_pred = np.array([100, 105, 115, 125])  # Also always up
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 100.0

    def test_wrong_direction(self):
        y_true = np.array([100, 110, 120])  # Up
        y_pred = np.array([100, 90, 80])    # Down
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 0.0

    def test_insufficient_data(self):
        y_true = np.array([100])
        y_pred = np.array([110])
        assert calculate_directional_accuracy(y_true, y_pred) == 0.0

    def test_mixed_directions(self):
        y_true = np.array([100, 110, 105, 115])  # Up, Down, Up
        y_pred = np.array([100, 108, 102, 120])   # Up, Down, Up
        da = calculate_directional_accuracy(y_true, y_pred)
        assert da == 100.0  # All 3 directions correct


class TestCalculateMetrics:
    def test_returns_all_metrics(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        metrics = calculate_metrics(y_true, y_pred, model_name="TestModel")
        
        assert metrics['Model'] == 'TestModel'
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE (%)' in metrics
        assert 'R²' in metrics
        assert 'SMAPE (%)' in metrics
        assert 'Directional Accuracy (%)' in metrics

    def test_dimension_mismatch_raises(self):
        y_true = np.array([100, 200])
        y_pred = np.array([100, 200, 300])
        with pytest.raises(ValueError, match="tidak sama"):
            calculate_metrics(y_true, y_pred)

    def test_perfect_prediction_metrics(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['RMSE'] == 0.0
        assert metrics['MAE'] == 0.0
        assert metrics['MAPE (%)'] == 0.0
        assert metrics['R²'] == 1.0

    def test_handles_2d_arrays(self):
        y_true = np.array([[100], [200], [300]])
        y_pred = np.array([[110], [190], [310]])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics['RMSE'] > 0


class TestCompareModels:
    def test_compare_empty_list(self):
        result = compare_models([])
        assert result['best_model'] is None
        assert result['table'] == []

    def test_compare_two_models(self):
        metrics_a = calculate_metrics(
            np.array([100, 200, 300]), 
            np.array([110, 190, 310]), 
            model_name="ModelA"
        )
        metrics_b = calculate_metrics(
            np.array([100, 200, 300]), 
            np.array([120, 180, 320]),  # Worse predictions
            model_name="ModelB"
        )
        
        result = compare_models([metrics_a, metrics_b])
        assert result['best_model'] == 'ModelA'  # Lower MAPE
        assert result['table'][0]['Model'] == 'ModelA'  # First in sorted list

    def test_single_model(self):
        metrics = calculate_metrics(
            np.array([100, 200]), np.array([100, 200]), model_name="Only"
        )
        result = compare_models([metrics])
        assert result['best_model'] == 'Only'


class TestPrintReport:
    def test_print_does_not_crash(self, capsys):
        metrics = calculate_metrics(
            np.array([100, 200, 300]),
            np.array([110, 190, 310]),
            model_name="TestPrint"
        )
        print_evaluation_report(metrics)
        captured = capsys.readouterr()
        assert "TestPrint" in captured.out
        assert "RMSE" in captured.out
