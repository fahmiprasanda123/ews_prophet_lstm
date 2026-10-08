"""
Protokol evaluasi terpadu untuk Agri-AI EWS (dipakai Model Laboratory dan skrip tesis).

Ringkasan protokol
------------------
1. Deret harga satu kombinasi komoditas-provinsi diubah ke kalender harian.
   Hari tanpa catatan diisi forward-fill (maks. `ffill_limit` hari); celah yang
   lebih panjang diinterpolasi linear. Jumlah hari yang diisi dicatat.
2. Split kronologis 80/20: 80% hari pertama = data latih, 20% terakhir = data uji.
3. Rolling-origin: periode uji dibagi menjadi jendela berurutan sepanjang
   `horizon` hari (bawaan 30). Pada setiap titik asal, model hanya boleh memakai
   data SEBELUM titik asal tersebut dan memprediksi `horizon` hari ke depan.
   - Naive Seasonal, SMA-30: dihitung dari data sebelum titik asal.
   - ARIMA dan Prophet: dipasang ulang pada setiap titik asal (expanding window).
   - BiLSTM dan TFT: dilatih SEKALI pada data latih, lalu menerima riwayat harga
     aktual sampai titik asal sebagai input (BiLSTM rekursif, TFT multi-horizon).
4. Bobot Smart Ensemble dicari dengan pencarian grid (kelipatan 5%) yang
   meminimalkan MAPE pada periode validasi = 20% terakhir data latih. Untuk itu
   model penyusun dilatih ulang pada data sebelum periode validasi dan dievaluasi
   dengan protokol rolling-origin yang sama. Bila sebaran prediksi antarmodel pada
   suatu tanggal terlalu besar (koefisien variasi > ambang), ensemble memakai
   model dengan MAPE validasi terbaik (mekanisme fallback).
5. Kovariat iklim (opsional) diperlakukan ex-ante: untuk hari sesudah titik asal,
   curah hujan dan suhu (rata-rata 30 hari) diganti klimatologi bulanan dari data
   sebelum titik asal, dan indeks ENSO memakai nilai terakhir yang sudah tersedia
   (jeda publikasi 30 hari). Tidak ada nilai kovariat masa depan yang bocor.
6. Semua prediksi disejajarkan per TANGGAL sebelum metrik dihitung.
   Directional Accuracy dihitung relatif terhadap harga terakhir di titik asal.
"""
from __future__ import annotations

import copy
import logging
import time
import warnings

import numpy as np
import pandas as pd

from models.evaluation import calculate_metrics, calculate_mape, mape_category

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.2,          # fraksi akhir data latih untuk validasi bobot ensemble
    'horizon': 30,
    'ffill_limit': 7,
    'seasonal_period': 365,
    'sma_window': 30,
    'arima_order': (5, 1, 0),
    'prophet': {
        'changepoint_prior_scale': 0.05,
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'holidays_country': 'ID',
    },
    'lstm': {
        'seq_length': 30, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2,
        'epochs': 50, 'batch_size': 64, 'lr': 1e-3, 'patience': 5,
        'val_fraction': 0.1, 'mc_samples': 50,
    },
    'tft': {
        'max_encoder_length': 90, 'max_epochs': 15, 'batch_size': 64,
        'learning_rate': 1e-3, 'hidden_size': 32, 'attention_head_size': 2, 'dropout': 0.1,
    },
    'ensemble': {'grid_step': 0.05, 'metric': 'MAPE (%)', 'fallback_cv': 0.15},
    'enso_lag_days': 30,
    'seed': 42,
}

MODEL_ORDER = ['Naive Seasonal', 'SMA-30', 'ARIMA(5,1,0)', 'Prophet', 'BiLSTM', 'TFT', 'Smart Ensemble']
MODEL_KEYS = {
    'naive': 'Naive Seasonal', 'sma': 'SMA-30', 'arima': 'ARIMA(5,1,0)', 'prophet': 'Prophet',
    'bilstm': 'BiLSTM', 'tft': 'TFT', 'ensemble': 'Smart Ensemble',
}
ALL_MODELS = list(MODEL_KEYS.keys())
CONSTITUENTS = ['Prophet', 'BiLSTM', 'TFT']
PROPHET_REGRESSORS = ['rainfall_30d', 'temperature_30d', 'enso_avail']
TFT_KNOWN_REALS = ['enso_avail']
TFT_UNKNOWN_REALS = ['rainfall_30d', 'temperature_30d']
HORIZON_BUCKETS = [(1, 7, '1–7 hari'), (8, 14, '8–14 hari'), (15, 30, '15–30 hari')]


# ---------------------------------------------------------------------- #
# Utilitas umum
# ---------------------------------------------------------------------- #
def merge_config(config: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def quiet_libraries():
    """Matikan log verbose Prophet/cmdstanpy/Lightning agar keluaran bersih."""
    for name in ['cmdstanpy', 'prophet', 'lightning', 'lightning.pytorch', 'pytorch_lightning',
                 'pytorch_forecasting', 'lightning.fabric']:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')


def load_frozen_data(csv_path: str, end_date: str | None = None) -> pd.DataFrame:
    """Muat snapshot data harga (CSV) yang dibekukan untuk evaluasi."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    if end_date:
        df = df[df['date'] <= pd.Timestamp(end_date)]
    return df.sort_values(['province', 'commodity', 'date']).reset_index(drop=True)


def prepare_series(df: pd.DataFrame, province: str, commodity: str, ffill_limit: int = 7):
    """Deret harga harian kontinu untuk satu kombinasi + ringkasan penanganan celah."""
    sub = df[(df['province'] == province) & (df['commodity'] == commodity)][['date', 'price']].dropna()
    if sub.empty:
        raise ValueError(f"Tidak ada data untuk {commodity} di {province}.")
    raw = sub.groupby('date')['price'].mean().sort_index()
    calendar = pd.date_range(raw.index.min(), raw.index.max(), freq='D')
    y = raw.reindex(calendar)
    missing = y.isna()
    filled = y.ffill(limit=ffill_limit)
    n_ffill = int((missing & filled.notna()).sum())
    still_missing = filled.isna()
    filled = filled.interpolate(limit_direction='both')
    gap_lengths = (missing != missing.shift()).cumsum()[missing].value_counts()
    info = {
        'province': province,
        'commodity': commodity,
        'n_records': int(len(raw)),
        'start': calendar[0].date().isoformat(),
        'end': calendar[-1].date().isoformat(),
        'n_days': int(len(calendar)),
        'n_missing_days': int(missing.sum()),
        'n_ffilled': n_ffill,
        'n_interpolated': int(still_missing.sum()),
        'max_gap_days': int(gap_lengths.max()) if len(gap_lengths) else 0,
    }
    filled.name = 'price'
    return filled.astype(float), info


def make_windows(start_idx: int, end_idx: int, horizon: int) -> list[dict]:
    """Jendela rolling-origin yang menutup [start_idx, end_idx) tanpa tumpang tindih.

    Jendela terakhir yang lebih pendek dari horizon digeser mundur agar panjang
    prakiraannya tetap = horizon; hanya hari yang belum tercakup yang dipakai.
    Setiap jendela: origin (indeks hari pertama yang diprediksi), keep_from, end.
    """
    windows, keep = [], start_idx
    while keep < end_idx:
        end = min(keep + horizon, end_idx)
        origin = keep if end - keep == horizon else max(end_idx - horizon, 0)
        windows.append({'origin': origin, 'keep_from': keep, 'end': end})
        keep = end
    return windows


def _assemble(y: pd.Series, windows: list[dict], window_preds: list, lowers=None, uppers=None):
    rows = []
    for i, (w, pred) in enumerate(zip(windows, window_preds)):
        o, k, e = w['origin'], w['keep_from'], w['end']
        pred = np.asarray(pred, dtype=float)
        lo = np.asarray(lowers[i], dtype=float) if lowers is not None else None
        up = np.asarray(uppers[i], dtype=float) if uppers is not None else None
        for t in range(k, e):
            rows.append({
                'date': y.index[t], 'pred': pred[t - o],
                'lower': lo[t - o] if lo is not None else np.nan,
                'upper': up[t - o] if up is not None else np.nan,
                'origin_date': y.index[o], 'horizon': t - o + 1, 'y_origin': y.iloc[o - 1],
            })
    return pd.DataFrame(rows).set_index('date')


# ---------------------------------------------------------------------- #
# Kovariat iklim
# ---------------------------------------------------------------------- #
def load_climate_covariates(province: str, start, end, cache_dir: str = 'data/cache',
                            strict: bool = True) -> pd.DataFrame:
    """Curah hujan & suhu harian (Open-Meteo) dan ONI (NOAA CPC) untuk satu provinsi.

    Hasil disimpan sebagai CSV di `cache_dir` agar analisis dapat direproduksi
    tanpa mengunduh ulang. Dengan strict=True, kegagalan unduh menghentikan proses
    (tidak ada data sintetis).
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    slug = province.lower().replace(' ', '_')
    path = os.path.join(cache_dir, f"iklim_{slug}_{start:%Y%m%d}_{end:%Y%m%d}.csv")
    if os.path.exists(path):
        cached = pd.read_csv(path, parse_dates=['date']).set_index('date')
        return cached
    from data.weather_client import WeatherClient
    client = WeatherClient(use_live=True, strict=strict)
    dates = pd.date_range(start, end, freq='D')
    raw = pd.DataFrame({
        'rainfall_mm': client.get_rainfall(province, dates).values,
        'temperature_c': client.get_temperature(province, dates).values,
        'enso_index': client.get_enso_index(dates).values,
    }, index=dates)
    raw.index.name = 'date'
    raw.to_csv(path)
    return raw


def derive_covariates(raw: pd.DataFrame, index: pd.DatetimeIndex, enso_lag_days: int = 30) -> pd.DataFrame:
    """Turunkan kovariat model: rata-rata 30 hari curah hujan & suhu, ENSO tersedia (berjeda)."""
    raw = raw.sort_index()
    cov = pd.DataFrame(index=raw.index)
    cov['rainfall_30d'] = raw['rainfall_mm'].rolling(30, min_periods=1).mean()
    cov['temperature_30d'] = raw['temperature_c'].rolling(30, min_periods=1).mean()
    cov['enso_avail'] = raw['enso_index'].shift(enso_lag_days).bfill()
    return cov.reindex(index).ffill().bfill()


def exante_covariates(cov: pd.DataFrame, origin_date, future_dates) -> pd.DataFrame:
    """Nilai kovariat untuk hari >= titik asal tanpa memakai informasi masa depan."""
    hist = cov.loc[:pd.Timestamp(origin_date) - pd.Timedelta(days=1)]
    out = pd.DataFrame(index=pd.DatetimeIndex(future_dates))
    for col in ['rainfall_30d', 'temperature_30d']:
        if col in cov.columns:
            clim = hist.groupby(hist.index.month)[col].mean()
            out[col] = [clim.get(d.month, hist[col].mean()) for d in out.index]
    if 'enso_avail' in cov.columns:
        out['enso_avail'] = float(hist['enso_avail'].iloc[-1])
    return out


# ---------------------------------------------------------------------- #
# Model baseline
# ---------------------------------------------------------------------- #
def run_naive_seasonal(y, windows, period=365):
    vals = y.values
    preds = []
    for w in windows:
        o, e = w['origin'], w['end']
        preds.append([vals[t - period] if t - period >= 0 else vals[o - 1] for t in range(o, e)])
    return _assemble(y, windows, preds)


def run_sma(y, windows, window=30):
    vals = y.values
    preds = [np.full(w['end'] - w['origin'], vals[w['origin'] - window:w['origin']].mean()) for w in windows]
    return _assemble(y, windows, preds)


def run_arima(y, windows, order=(5, 1, 0)):
    from statsmodels.tsa.arima.model import ARIMA
    vals = y.values
    preds = []
    for w in windows:
        o, e = w['origin'], w['end']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = ARIMA(vals[:o], order=order, enforce_stationarity=False,
                           enforce_invertibility=False).fit(method_kwargs={'maxiter': 200})
            preds.append(np.clip(np.asarray(fitted.forecast(steps=e - o)), 0, None))
    return _assemble(y, windows, preds)


# ---------------------------------------------------------------------- #
# Model AI
# ---------------------------------------------------------------------- #
def _fit_prophet(train_y: pd.Series, cov_train: pd.DataFrame | None, params: dict):
    from prophet import Prophet
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    model = Prophet(
        yearly_seasonality=params.get('yearly_seasonality', True),
        weekly_seasonality=params.get('weekly_seasonality', True),
        daily_seasonality=False,
        changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
        interval_width=0.90,
    )
    if params.get('holidays_country'):
        model.add_country_holidays(country_name=params['holidays_country'])
    frame = pd.DataFrame({'ds': train_y.index, 'y': train_y.values})
    if cov_train is not None:
        for col in PROPHET_REGRESSORS:
            model.add_regressor(col)
            frame[col] = cov_train[col].values
    model.fit(frame)
    return model


def run_prophet(y, windows, cov=None, params=None):
    params = params or DEFAULT_CONFIG['prophet']
    preds, lowers, uppers = [], [], []
    for w in windows:
        o, e = w['origin'], w['end']
        model = _fit_prophet(y.iloc[:o], cov.iloc[:o] if cov is not None else None, params)
        future = pd.DataFrame({'ds': y.index[o:e]})
        if cov is not None:
            ex = exante_covariates(cov, y.index[o], y.index[o:e])
            for col in PROPHET_REGRESSORS:
                future[col] = ex[col].values
        fc = model.predict(future)
        preds.append(fc['yhat'].values)
        lowers.append(fc['yhat_lower'].values)
        uppers.append(fc['yhat_upper'].values)
    return _assemble(y, windows, preds, lowers, uppers)


def train_bilstm(y: pd.Series, train_end: int, params: dict, seed: int):
    """Latih BiLSTM sekali pada y[:train_end] (skaler juga hanya dari data latih)."""
    import torch
    from models.lstm_forecast import LSTMForecaster
    set_seed(seed)
    forecaster = LSTMForecaster(seq_length=params['seq_length'], hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'], dropout=params['dropout'])
    train_vals = y.values[:train_end].reshape(-1, 1)
    forecaster.scaler.fit(train_vals)
    scaled = forecaster.scaler.transform(train_vals)
    L = params['seq_length']
    X = np.stack([scaled[i:i + L] for i in range(len(scaled) - L)])
    Y = np.stack([scaled[i + L] for i in range(len(scaled) - L)])
    forecaster.train_single_series(torch.FloatTensor(X), torch.FloatTensor(Y), epochs=params['epochs'],
                                   lr=params['lr'], batch_size=params['batch_size'],
                                   val_fraction=params['val_fraction'], patience=params['patience'], seed=seed)
    return forecaster


def run_bilstm(y, windows, train_end, params=None, seed=42, with_mc=True):
    params = {**DEFAULT_CONFIG['lstm'], **(params or {})}
    forecaster = train_bilstm(y, train_end, params, seed)
    L = params['seq_length']
    preds, lowers, uppers = [], [], []
    for w in windows:
        o, e = w['origin'], w['end']
        history = y.values[o - L:o]
        preds.append(forecaster.predict_multi_step(history, steps=e - o))
        if with_mc:
            set_seed(seed + o)
            mc = forecaster.predict_with_uncertainty(history, steps=e - o, n_samples=params['mc_samples'])
            lowers.append(mc['lower'])
            uppers.append(mc['upper'])
    out = _assemble(y, windows, preds, lowers if with_mc else None, uppers if with_mc else None)
    out.attrs['epochs_trained'] = getattr(forecaster, 'epochs_trained', None)
    return out


def run_tft(y, windows, train_end, province, commodity, cov=None, horizon=30, params=None, seed=42):
    from models.tft_forecast import TFTForecaster, TFT_AVAILABLE
    if not TFT_AVAILABLE:
        raise RuntimeError("pytorch-forecasting belum terpasang; TFT tidak dapat dievaluasi.")
    params = {**DEFAULT_CONFIG['tft'], **(params or {})}
    set_seed(seed)
    tft = TFTForecaster(max_prediction_length=horizon, max_encoder_length=params['max_encoder_length'])
    frame = tft.build_frame(y, province, commodity, covariates=cov)
    known = TFT_KNOWN_REALS if cov is not None else []
    unknown = TFT_UNKNOWN_REALS if cov is not None else []
    dataset = tft.build_training_dataset(frame, training_cutoff=train_end - 1,
                                         known_reals=known, unknown_reals=unknown)
    tft.train(dataset, max_epochs=params['max_epochs'], batch_size=params['batch_size'],
              learning_rate=params['learning_rate'], hidden_size=params['hidden_size'],
              attention_head_size=params['attention_head_size'], dropout=params['dropout'],
              seed=seed, quiet=True)
    preds, lowers, uppers = [], [], []
    for w in windows:
        o, e = w['origin'], w['end']
        if e - o != horizon:
            raise ValueError("Jendela TFT harus sepanjang horizon.")
        sub = frame.iloc[:e].copy()
        if cov is not None:
            ex = exante_covariates(cov, y.index[o], y.index[o:e])
            for col in ex.columns:
                sub.loc[sub.index[o:e], col] = ex[col].values
        res = tft.predict(sub, dataset)
        preds.append(res['mean'][-horizon:])
        lowers.append(res['lower'][-horizon:])
        uppers.append(res['upper'][-horizon:])
    return _assemble(y, windows, preds, lowers, uppers)


# ---------------------------------------------------------------------- #
# Evaluasi satu kombinasi
# ---------------------------------------------------------------------- #
def _model_metrics(name: str, pred_df: pd.DataFrame, y: pd.Series) -> dict:
    actual = y.reindex(pred_df.index).values
    pred = pred_df['pred'].values
    m = calculate_metrics(actual, pred, name, y_origin=pred_df['y_origin'].values)
    ape = np.abs(pred - actual) / np.abs(actual) * 100
    m['Kategori MAPE'] = mape_category(m['MAPE (%)'])
    m['Dalam ±5% (%)'] = float(np.mean(ape <= 5) * 100)
    m['Dalam ±10% (%)'] = float(np.mean(ape <= 10) * 100)
    return m


def horizon_mape(predictions: dict, y: pd.Series) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if name not in predictions:
            continue
        df = predictions[name]
        row = {'Model': name}
        for lo, hi, label in HORIZON_BUCKETS:
            part = df[(df['horizon'] >= lo) & (df['horizon'] <= hi)]
            row[label] = calculate_mape(y.reindex(part.index).values, part['pred'].values) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def interval_coverage(predictions: dict, y: pd.Series) -> pd.DataFrame:
    rows = []
    labels = {'Prophet': 'Interval 90% Prophet', 'BiLSTM': 'Interval 90% MC Dropout (50 sampel)',
              'TFT': 'Interval kuantil 2–98% TFT'}
    for name, label in labels.items():
        df = predictions.get(name)
        if df is None or df['lower'].isna().all():
            continue
        actual = y.reindex(df.index).values
        inside = (actual >= df['lower'].values) & (actual <= df['upper'].values)
        width = (df['upper'].values - df['lower'].values) / actual * 100
        rows.append({'Model': name, 'Interval': label, 'Cakupan aktual (%)': float(np.mean(inside) * 100),
                     'Lebar rata-rata (% harga)': float(np.mean(width))})
    return pd.DataFrame(rows)


def evaluate_series(df: pd.DataFrame, province: str, commodity: str, config: dict | None = None,
                    models: list | None = None, covariates_raw: pd.DataFrame | None = None,
                    progress=None) -> dict:
    """Jalankan protokol evaluasi lengkap untuk satu kombinasi komoditas-provinsi.

    Args:
        df: Data harga (date, province, commodity, price).
        config: Penimpaan konfigurasi (lihat DEFAULT_CONFIG).
        models: Subset kunci model (naive, sma, arima, prophet, bilstm, tft, ensemble).
        covariates_raw: DataFrame harian (rainfall_mm, temperature_c, enso_index) atau None.
        progress: Callable opsional progress(pesan: str, fraksi: float).
    """
    quiet_libraries()
    cfg = merge_config(config)
    models = models or ALL_MODELS
    report = progress or (lambda msg, frac: logger.info(msg))
    set_seed(cfg['seed'])

    y, info = prepare_series(df, province, commodity, cfg['ffill_limit'])
    n = len(y)
    split = int(n * (1 - cfg['test_size']))
    val_start = int(split * (1 - cfg['val_size']))
    H = cfg['horizon']
    test_w = make_windows(split, n, H)
    val_w = make_windows(val_start, split, H)
    cov = derive_covariates(covariates_raw, y.index, cfg['enso_lag_days']) if covariates_raw is not None else None

    preds, val_preds, timing, notes = {}, {}, {}, []
    need_val = 'ensemble' in models
    steps = [m for m in ['naive', 'sma', 'arima', 'prophet', 'bilstm', 'tft'] if m in models]

    for i, key in enumerate(steps):
        name = MODEL_KEYS[key]
        report(f"Menjalankan {name}…", i / max(len(steps) + 1, 1))
        t0 = time.time()
        try:
            if key == 'naive':
                preds[name] = run_naive_seasonal(y, test_w, cfg['seasonal_period'])
            elif key == 'sma':
                preds[name] = run_sma(y, test_w, cfg['sma_window'])
            elif key == 'arima':
                preds[name] = run_arima(y, test_w, tuple(cfg['arima_order']))
            elif key == 'prophet':
                preds[name] = run_prophet(y, test_w, cov, cfg['prophet'])
                if need_val:
                    val_preds[name] = run_prophet(y, val_w, cov, cfg['prophet'])
            elif key == 'bilstm':
                preds[name] = run_bilstm(y, test_w, split, cfg['lstm'], cfg['seed'])
                if need_val:
                    val_preds[name] = run_bilstm(y, val_w, val_start, cfg['lstm'], cfg['seed'], with_mc=False)
            elif key == 'tft':
                preds[name] = run_tft(y, test_w, split, province, commodity, cov, H, cfg['tft'], cfg['seed'])
                if need_val:
                    val_preds[name] = run_tft(y, val_w, val_start, province, commodity, cov, H,
                                              cfg['tft'], cfg['seed'])
        except Exception as exc:  # model gagal tidak menghentikan model lain
            notes.append(f"{name} gagal: {exc}")
            logger.warning(f"{name} gagal: {exc}")
        timing[name] = round(time.time() - t0, 1)

    ensemble_info = None
    if need_val and len(val_preds) >= 2:
        from models.ensemble import SmartEnsemble
        report("Mencari bobot Smart Ensemble…", len(steps) / (len(steps) + 1))
        ens = SmartEnsemble()
        ensemble_info = ens.fit_weights_grid({k: v['pred'] for k, v in val_preds.items()},
                                             y, step=cfg['ensemble']['grid_step'],
                                             metric=cfg['ensemble']['metric'])
        combined, flag = ens.combine_series({k: preds[k]['pred'] for k in val_preds if k in preds},
                                            fallback_cv=cfg['ensemble']['fallback_cv'])
        base = preds[next(k for k in val_preds if k in preds)]
        ens_df = base[['origin_date', 'horizon', 'y_origin']].copy()
        ens_df['pred'] = combined.reindex(base.index).values
        ens_df['lower'] = np.nan
        ens_df['upper'] = np.nan
        ens_df['fallback'] = flag.reindex(base.index).fillna(False).values
        preds['Smart Ensemble'] = ens_df
        ensemble_info['fallback_days'] = int(ens_df['fallback'].sum())
        ensemble_info['fallback_rate'] = float(ens_df['fallback'].mean() * 100)
    elif need_val:
        notes.append("Smart Ensemble dilewati: minimal dua model penyusun harus berhasil.")

    metrics = pd.DataFrame([_model_metrics(name, preds[name], y) for name in MODEL_ORDER if name in preds])
    report("Selesai.", 1.0)
    return {
        'province': province, 'commodity': commodity, 'series': y, 'data_info': info,
        'split': {'n_days': n, 'train_end': y.index[split - 1].date().isoformat(),
                  'test_start': y.index[split].date().isoformat(), 'test_end': y.index[-1].date().isoformat(),
                  'n_train': split, 'n_test': n - split,
                  'val_start': y.index[val_start].date().isoformat(), 'n_val': split - val_start,
                  'n_windows_test': len(test_w), 'n_windows_val': len(val_w)},
        'windows_test': test_w, 'predictions': preds, 'val_predictions': val_preds,
        'metrics': metrics, 'horizon_mape': horizon_mape(preds, y),
        'intervals': interval_coverage(preds, y), 'ensemble': ensemble_info,
        'covariates_used': cov is not None, 'timing': timing, 'notes': notes, 'config': cfg,
    }


def predictions_frame(result: dict) -> pd.DataFrame:
    """Gabungkan semua prediksi uji menjadi satu tabel (tanggal × model)."""
    y = result['series']
    first = next(iter(result['predictions'].values()))
    out = pd.DataFrame({'aktual': y.reindex(first.index)})
    out['titik_asal'] = first['origin_date']
    out['horizon'] = first['horizon']
    for name in MODEL_ORDER:
        if name in result['predictions']:
            out[name] = result['predictions'][name]['pred']
    out.index.name = 'tanggal'
    return out


# ---------------------------------------------------------------------- #
# Analisis pendukung (deterministik)
# ---------------------------------------------------------------------- #
def feature_importance(df: pd.DataFrame, province: str, commodity: str) -> pd.DataFrame:
    """Korelasi Pearson absolut fitur prediktor terhadap harga (sama dengan Model Laboratory)."""
    series = df[(df['province'] == province) & (df['commodity'] == commodity)].sort_values('date')
    price = series['price'].values.astype(float)
    rows = [
        ('Lag 1 hari', 'Autolag harga', abs(np.corrcoef(price[1:], price[:-1])[0, 1])),
        ('Lag 7 hari', 'Autolag harga', abs(np.corrcoef(price[7:], price[:-7])[0, 1])),
        ('Lag 14 hari', 'Autolag harga', abs(np.corrcoef(price[14:], price[:-14])[0, 1])),
        ('Lag 30 hari', 'Autolag harga', abs(np.corrcoef(price[30:], price[:-30])[0, 1])),
        ('Bulan', 'Temporal', abs(series['date'].dt.month.corr(series['price']))),
        ('Hari dalam seminggu', 'Temporal', abs(series['date'].dt.dayofweek.corr(series['price']))),
    ]
    pivot = df[df['province'] == province].pivot_table(index='date', columns='commodity', values='price')
    for other in pivot.columns:
        if other != commodity and commodity in pivot.columns:
            value = pivot[commodity].corr(pivot[other])
            if not np.isnan(value):
                rows.append((f'Korelasi dengan {other}', 'Lintas komoditas', abs(value)))
    out = pd.DataFrame(rows, columns=['Fitur', 'Kelompok', '|r|'])
    return out.sort_values('|r|', ascending=False).reset_index(drop=True)


def climate_correlation(y: pd.Series, covariates_raw: pd.DataFrame, lags=(0, 7, 14, 30)) -> pd.DataFrame:
    """Korelasi Pearson & Spearman kovariat iklim (t - lag) dengan harga (t) dan perubahan harga 30 hari."""
    from scipy.stats import pearsonr, spearmanr
    raw = covariates_raw.reindex(y.index).ffill()
    variables = {
        'Curah hujan (rata-rata 30 hari)': raw['rainfall_mm'].rolling(30, min_periods=1).mean(),
        'Suhu (rata-rata 30 hari)': raw['temperature_c'].rolling(30, min_periods=1).mean(),
        'Indeks ENSO (ONI)': raw['enso_index'],
    }
    targets = {'Harga': y, 'Perubahan harga 30 hari (%)': y.pct_change(30) * 100}
    rows = []
    for tname, target in targets.items():
        for vname, var in variables.items():
            for lag in lags:
                pair = pd.concat([target, var.shift(lag)], axis=1).dropna()
                r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                rho, p_s = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rows.append({'Target': tname, 'Kovariat': vname, 'Lag (hari)': lag, 'Pearson r': r,
                             'p (Pearson)': p, 'Spearman ρ': rho, 'p (Spearman)': p_s, 'n': len(pair)})
    return pd.DataFrame(rows)


def data_summary(df: pd.DataFrame, ffill_limit: int = 7) -> tuple[dict, pd.DataFrame]:
    """Ringkasan pemeriksaan dan pembersihan data untuk seluruh kombinasi."""
    total = {
        'Jumlah catatan': int(len(df)),
        'Tanggal awal': df['date'].min().date().isoformat(),
        'Tanggal akhir': df['date'].max().date().isoformat(),
        'Jumlah komoditas': int(df['commodity'].nunique()),
        'Jumlah provinsi': int(df['province'].nunique()),
        'Catatan duplikat (tanggal-provinsi-komoditas)': int(df.duplicated(['date', 'province', 'commodity']).sum()),
        'Harga kosong': int(df['price'].isna().sum()),
        'Harga nol/negatif': int((df['price'] <= 0).sum()),
    }
    rows = []
    for (prov, comm), _ in df.groupby(['province', 'commodity']):
        _, info = prepare_series(df, prov, comm, ffill_limit)
        rows.append(info)
    per_series = pd.DataFrame(rows)
    total['Jumlah deret (komoditas × provinsi)'] = int(len(per_series))
    total['Total hari kalender seluruh deret'] = int(per_series['n_days'].sum())
    total['Hari tanpa catatan'] = int(per_series['n_missing_days'].sum())
    total['Diisi forward-fill (≤ %d hari)' % ffill_limit] = int(per_series['n_ffilled'].sum())
    total['Diinterpolasi linear (celah > %d hari)' % ffill_limit] = int(per_series['n_interpolated'].sum())
    total['Celah terpanjang (hari)'] = int(per_series['max_gap_days'].max())
    return total, per_series
