import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Menghitung dan mengembalikan berbagai metrik evaluasi: RMSE, MAE, dan MAPE.
    
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
    
    metrics = {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape
    }
    
    return metrics

def print_evaluation_report(metrics):
    """
    Mencetak laporan hasil evaluasi metrik ke terminal dalam format tabel/rapi.
    
    Args:
        metrics (dict): Dictionary hasil output dari `calculate_metrics`.
    """
    print("-" * 40)
    print(f"📊 Laporan Evaluasi Eksperimen: {metrics.get('Model', 'Unknown')}")
    print("-" * 40)
    print(f"  Root Mean Squared Error (RMSE) : {metrics['RMSE']:,.2f}")
    print(f"  Mean Absolute Error (MAE)      : {metrics['MAE']:,.2f}")
    print(f"  Mean Abs Percentage Err (MAPE) : {metrics['MAPE (%)']:.2f}%")
    print("-" * 40)
