"""
Prakiraan operasional ke depan (setelah tanggal data terakhir).

Berbeda dengan protokol evaluasi (yang menyembunyikan data uji), di sini seluruh data
yang tersedia dipakai untuk melatih model, lalu harga diprakirakan 1–30 hari ke depan.
Horizon dibatasi 30 hari karena hanya horizon itu yang telah divalidasi (Sub-bab 3.6.2).
Bobot Smart Ensemble diambil dari hasil pencarian grid pada periode validasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluation_protocol import (
    DEFAULT_CONFIG, PROPHET_REGRESSORS, TFT_KNOWN_REALS, TFT_UNKNOWN_REALS, _fit_prophet, derive_covariates,
    exante_covariates, merge_config, prepare_series, quiet_libraries, set_seed, train_bilstm,
)

BULAN = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']


def _tgl(d):
    return f'{d.day} {BULAN[d.month - 1]} {d.year}'


def forecast_path(df, province, commodity, horizon=30, config=None, covariates_raw=None, weights=None,
                  best_single=None, progress=None):
    """Prakiraan harian `horizon` hari setelah tanggal data terakhir.

    Returns:
        dict: path (DataFrame per tanggal: tiap model, Smart Ensemble, batas bawah/atas, fallback),
        last_date, last_price, weights, covariates_used, province, commodity.
    """
    quiet_libraries()
    cfg = merge_config(config)
    H = int(horizon)
    if not 1 <= H <= cfg['horizon']:
        raise ValueError(f"Horizon harus 1–{cfg['horizon']} hari (horizon yang divalidasi).")
    report = progress or (lambda msg, frac: None)
    set_seed(cfg['seed'])
    y, info = prepare_series(df, province, commodity, cfg['ffill_limit'])
    vals, n = y.values, len(y)
    full = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=cfg['horizon'], freq='D')
    idx = full[:H]
    cov = derive_covariates(covariates_raw, y.index, cfg['enso_lag_days']) if covariates_raw is not None else None
    cov_full = exante_covariates(cov, full[0], full) if cov is not None else None
    out = pd.DataFrame(index=idx)
    bounds = {}

    # Metode konvensional (pembanding)
    p = cfg['seasonal_period']
    naive = []
    for i in range(H):
        lag = n + i - p
        while lag >= n:
            lag -= p
        naive.append(vals[lag] if lag >= 0 else vals[-1])
    out['Naive Seasonal'] = naive
    out['SMA-30'] = float(vals[-cfg['sma_window']:].mean())
    try:
        import warnings
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fit = ARIMA(vals, order=tuple(cfg['arima_order']), enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        out['ARIMA(5,1,0)'] = np.clip(np.asarray(fit.forecast(steps=H)), 0, None)
    except Exception:
        pass

    # Prophet
    report('Melatih Prophet…', 0.1)
    model = _fit_prophet(y, cov, cfg['prophet'])
    fut = pd.DataFrame({'ds': idx})
    if cov is not None:
        for col in PROPHET_REGRESSORS:
            fut[col] = cov_full[col].values[:H]
    fc = model.predict(fut)
    out['Prophet'] = fc['yhat'].values
    bounds['Prophet'] = (fc['yhat_lower'].values, fc['yhat_upper'].values)

    # BiLSTM (dilatih pada seluruh data, prakiraan rekursif + MC Dropout)
    report('Melatih BiLSTM…', 0.35)
    lp = {**DEFAULT_CONFIG['lstm'], **cfg['lstm']}
    f = train_bilstm(y, n, lp, cfg['seed'])
    hist = vals[-lp['seq_length']:]
    out['BiLSTM'] = f.predict_multi_step(hist, steps=H)
    mc = f.predict_with_uncertainty(hist, steps=H, n_samples=lp['mc_samples'])
    bounds['BiLSTM'] = (np.asarray(mc['lower']), np.asarray(mc['upper']))

    # TFT (prakiraan multi-horizon ke depan)
    report('Melatih TFT…', 0.6)
    try:
        from models.tft_forecast import TFT_AVAILABLE, TFTForecaster
        if TFT_AVAILABLE:
            tp = {**DEFAULT_CONFIG['tft'], **cfg['tft']}
            tft = TFTForecaster(max_prediction_length=cfg['horizon'], max_encoder_length=tp['max_encoder_length'])
            frame = tft.build_frame(y, province, commodity, covariates=cov)
            dataset = tft.build_training_dataset(frame, training_cutoff=n - 1,
                                                 known_reals=TFT_KNOWN_REALS if cov is not None else [],
                                                 unknown_reals=TFT_UNKNOWN_REALS if cov is not None else [])
            tft.train(dataset, max_epochs=tp['max_epochs'], batch_size=tp['batch_size'],
                      learning_rate=tp['learning_rate'], hidden_size=tp['hidden_size'],
                      attention_head_size=tp['attention_head_size'], dropout=tp['dropout'],
                      seed=cfg['seed'], quiet=True)
            res = tft.forecast_future(frame, dataset, future_covariates=cov_full)
            out['TFT'] = np.asarray(res['mean'])[:H]
            bounds['TFT'] = (np.asarray(res['lower'])[:H], np.asarray(res['upper'])[:H])
    except Exception:
        pass

    # Smart Ensemble dengan bobot hasil validasi (+ fallback)
    report('Menggabungkan…', 0.9)
    if weights:
        from models.ensemble import SmartEnsemble
        ens = SmartEnsemble()
        ens.weights = {k: float(v) for k, v in weights.items() if k in out.columns}
        ens.best_single = best_single if best_single in ens.weights else max(ens.weights, key=ens.weights.get)
        combined, flag = ens.combine_series({k: out[k] for k in ens.weights},
                                            fallback_cv=cfg['ensemble']['fallback_cv'])
        out['Smart Ensemble'] = combined.reindex(idx).values
        out['fallback'] = flag.reindex(idx).fillna(False).values
        pos = {k: w for k, w in ens.weights.items() if w > 0 and k in bounds}
        if pos:
            tot = sum(pos.values())
            out['Batas bawah'] = sum(w * bounds[k][0] for k, w in pos.items()) / tot
            out['Batas atas'] = sum(w * bounds[k][1] for k, w in pos.items()) / tot
    out.index.name = 'tanggal'
    report('Selesai.', 1.0)
    return {'path': out, 'last_date': y.index[-1], 'last_price': float(vals[-1]), 'weights': weights,
            'covariates_used': cov is not None, 'province': province, 'commodity': commodity,
            'data_info': info}


def aggregate_forecast(path, freq='D'):
    """Ringkasan prakiraan: 'D' harian, 'W' mingguan (Senin–Minggu), 'M' bulanan (bulan kalender)."""
    cols = [c for c in path.columns if c != 'fallback']
    if freq == 'D':
        out = path[cols].copy()
        out.insert(0, 'Periode', [_tgl(d) for d in out.index])
        return out.reset_index(drop=True)
    rule = {'W': 'W-SUN', 'M': 'MS'}[freq]
    rows = []
    for _, part in path[cols].groupby(pd.Grouper(freq=rule)):
        if part.empty:
            continue
        a, b = part.index.min(), part.index.max()
        label = _tgl(a) if a == b else (f'{a.day}–{_tgl(b)}' if (a.month, a.year) == (b.month, b.year)
                                        else f'{_tgl(a)} – {_tgl(b)}')
        rows.append({'Periode': label, 'Jumlah hari': len(part), **part.mean().to_dict()})
    return pd.DataFrame(rows)
