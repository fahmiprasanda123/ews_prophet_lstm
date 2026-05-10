"""
Model evaluation metrics for Agri-AI EWS.
Supports RMSE, MAE, MAPE, R², SMAPE, and Directional Accuracy.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mape(y_true, y_pred):
    """
    Menghitung Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menghindari pembagian oleh nol
    non_zero_idx = y_true != 0
    if not np.any(non_zero_idx):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def calculate_smape(y_true, y_pred):
    """
    Menghitung Symmetric Mean Absolute Percentage Error (SMAPE).
    Range: 0-200%, lebih robust dibanding MAPE untuk nilai mendekati nol.
    """
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

def calculate_directional_accuracy(y_true, y_pred):
    """
    Menghitung Directional Accuracy (DA).
    Mengukur seberapa sering model benar dalam memprediksi arah perubahan harga.
    
    Returns:
        float: Persentase (0-100) prediksi arah yang benar.
    """
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    if len(y_true) < 2:
        return 0.0
    
    true_dir = np.diff(y_true)
    pred_dir = np.diff(y_pred)
    
    correct = np.sum(np.sign(true_dir) == np.sign(pred_dir))
    return (correct / len(true_dir)) * 100

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Menghitung dan mengembalikan berbagai metrik evaluasi:
    RMSE, MAE, MAPE, R², SMAPE, dan Directional Accuracy.
    
    Args:
        y_true (array-like): Nilai aktual.
        y_pred (array-like): Nilai prediksi.
        model_name (str): Nama model untuk keperluan log.
        
    Returns:
        dict: Dictionary berisi metrik evaluasi.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimensi label aktual ({len(y_true)}) dan prediksi ({len(y_pred)}) tidak sama.")
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    da = calculate_directional_accuracy(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R²": r2,
        "SMAPE (%)": smape,
        "Directional Accuracy (%)": da,
    }
    
    return metrics

def compare_models(model_metrics_list: list) -> dict:
    """
    Membandingkan beberapa model berdasarkan metrik evaluasi.
    
    Args:
        model_metrics_list: List of dicts from calculate_metrics().
    
    Returns:
        dict with 'table' (list of dicts) and 'best_model' (name of best by MAPE).
    """
    if not model_metrics_list:
        return {'table': [], 'best_model': None}
    
    # Sort by MAPE (lowest is best)
    sorted_models = sorted(model_metrics_list, key=lambda x: x.get('MAPE (%)', float('inf')))
    
    return {
        'table': sorted_models,
        'best_model': sorted_models[0].get('Model', 'Unknown'),
        'best_mape': sorted_models[0].get('MAPE (%)', None),
    }

def print_evaluation_report(metrics):
    """
    Mencetak laporan hasil evaluasi metrik ke terminal dalam format tabel/rapi.
    
    Args:
        metrics (dict): Dictionary hasil output dari `calculate_metrics`.
    """
    print("-" * 50)
    print(f"📊 Laporan Evaluasi Eksperimen: {metrics.get('Model', 'Unknown')}")
    print("-" * 50)
    print(f"  Root Mean Squared Error (RMSE)      : {metrics['RMSE']:,.2f}")
    print(f"  Mean Absolute Error (MAE)            : {metrics['MAE']:,.2f}")
    print(f"  Mean Abs Percentage Err (MAPE)       : {metrics['MAPE (%)']:.2f}%")
    print(f"  R² Score                             : {metrics['R²']:.4f}")
    print(f"  Symmetric MAPE (SMAPE)               : {metrics['SMAPE (%)']:.2f}%")
    print(f"  Directional Accuracy                 : {metrics['Directional Accuracy (%)']:.1f}%")
    print("-" * 50)
