#!/usr/bin/env python3
"""
Analisis sensitivitas parameter tiap algoritma (untuk Sub-bab 4.10 dan Pembahasan).

Setiap parameter diubah satu per satu (one-at-a-time) dari nilai bawaan, sementara
parameter lain tetap. Kinerja diukur pada periode uji dengan protokol yang sama dengan
Bab IV (rolling-origin 30 hari, data beku, kovariat iklim ex-ante). Nilai bawaan TIDAK
dipilih berdasarkan data uji; analisis ini menilai seberapa peka kinerja terhadap
pilihan parameter.

Contoh:
    python scripts/analisis_parameter.py                       # lengkap (±45–60 menit)
    python scripts/analisis_parameter.py --model BiLSTM,TFT    # sebagian algoritma
    python scripts/analisis_parameter.py --tanpa-kovariat --cepat   # uji alur saja
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.evaluation_protocol import (  # noqa: E402
    _model_metrics, derive_covariates, load_climate_covariates, load_frozen_data, make_windows, merge_config,
    prepare_series, quiet_libraries, run_arima, run_bilstm, run_naive_seasonal, run_prophet, run_sma, run_tft,
)

SWEEPS = [
    ('Naive Seasonal', 'Periode musiman (hari)', 'seasonal_period', [7, 30, 365], 365),
    ('SMA', 'Jendela rata-rata (hari)', 'sma_window', [7, 14, 30, 60], 30),
    ('ARIMA', 'Orde (p, d, q)', 'arima_order', [(1, 1, 0), (2, 1, 0), (5, 1, 0), (1, 1, 1), (2, 1, 2)], (5, 1, 0)),
    ('Prophet', 'changepoint_prior_scale', 'changepoint_prior_scale', [0.01, 0.05, 0.1, 0.5], 0.05),
    ('Prophet', 'Efek hari libur nasional', 'holidays_country', ['ID', None], 'ID'),
    ('Prophet', 'Kovariat iklim', 'kovariat', [True, False], True),
    ('BiLSTM', 'Panjang sekuens (hari)', 'seq_length', [14, 30, 60], 30),
    ('BiLSTM', 'Ukuran tersembunyi', 'hidden_size', [32, 64, 128], 128),
    ('BiLSTM', 'Dropout', 'dropout', [0.1, 0.2, 0.3], 0.2),
    ('TFT', 'Panjang encoder (hari)', 'max_encoder_length', [30, 60, 90], 90),
    ('TFT', 'Ukuran tersembunyi', 'hidden_size', [16, 32, 64], 32),
    ('TFT', 'Jumlah epoch', 'max_epochs', [5, 15, 30], 15),
    ('TFT', 'Kovariat iklim', 'kovariat', [True, False], True),
    ('Smart Ensemble', 'Langkah grid bobot', 'grid_step', [0.05, 0.10, 0.25], 0.05),
    ('Smart Ensemble', 'Ambang fallback (KV)', 'fallback_cv', [None, 0.10, 0.15, 0.20], 0.15),
]
QUICK = {'lstm': {'epochs': 3, 'mc_samples': 10}, 'tft': {'max_epochs': 1}}


def fmt_val(v):
    if v is None:
        return 'tidak dipakai'
    if v is True:
        return 'ya'
    if v is False:
        return 'tidak'
    if isinstance(v, tuple):
        return '(' + ', '.join(map(str, v)) + ')'
    return str(v).replace('.', ',')


def angka(x, d=2):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else \
        f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(ROOT, 'food_prices_real.csv'))
    ap.add_argument('--provinsi', default='Jawa Timur')
    ap.add_argument('--komoditas', default='Bawang Merah')
    ap.add_argument('--model', default='', help='Subset dipisah koma, mis. "BiLSTM,TFT".')
    ap.add_argument('--tanpa-kovariat', action='store_true')
    ap.add_argument('--cepat', action='store_true')
    ap.add_argument('--output', default=os.path.join(ROOT, 'parameter_outputs'))
    ap.add_argument('--lanjut', action='store_true',
                    help='Lanjutkan run yang terhenti: baris yang sudah ada di CSV tidak dihitung ulang.')
    a = ap.parse_args()
    quiet_libraries()
    os.makedirs(a.output, exist_ok=True)
    import platform
    versi = {'python': platform.python_version()}
    for mod in ['numpy', 'pandas', 'statsmodels', 'prophet', 'torch', 'pytorch_forecasting']:
        try:
            versi[mod] = __import__(mod).__version__
        except Exception:
            versi[mod] = '-'
    print('Lingkungan:', sys.executable, versi, flush=True)
    ref_path = os.path.join(ROOT, 'thesis_outputs', '4_2_metrik_model.csv')
    ref = pd.read_csv(ref_path).set_index('Model')['MAPE (%)'] if os.path.exists(ref_path) else None
    REFNAME = {'Naive Seasonal': 'Naive Seasonal', 'SMA': 'SMA-30', 'ARIMA': 'ARIMA(5,1,0)', 'Prophet': 'Prophet',
               'BiLSTM': 'BiLSTM', 'TFT': 'TFT', 'Smart Ensemble': 'Smart Ensemble'}

    def cek_bawaan(df_rows, label):
        if ref is None or a.cepat or a.tanpa_kovariat or df_rows.empty:
            return True
        ok = True
        for _, r in df_rows[df_rows['Bawaan'].astype(str) == 'True'].drop_duplicates('Model').iterrows():
            target = float(ref.get(REFNAME[r['Model']], np.nan))
            beda = abs(float(r['MAPE (%)']) - target)
            status = 'cocok' if beda <= 0.01 else 'TIDAK COCOK'
            ok &= beda <= 0.01
            print(f'  [{label}] {r["Model"]:15s} MAPE bawaan {float(r["MAPE (%)"]):.4f} vs Tabel 4.2 {target:.4f} -> {status}')
        return ok
    base = merge_config(QUICK if a.cepat else {})
    df = load_frozen_data(a.data)
    y, _ = prepare_series(df, a.provinsi, a.komoditas, base['ffill_limit'])
    n, H = len(y), base['horizon']
    split = int(n * (1 - base['test_size']))
    val_start = int(split * (1 - base['val_size']))
    test_w, val_w = make_windows(split, n, H), make_windows(val_start, split, H)

    cov = None
    if not a.tanpa_kovariat:
        try:
            raw = load_climate_covariates(a.provinsi, y.index.min(), y.index.max(),
                                          cache_dir=os.path.join(ROOT, 'data', 'cache'), strict=True)
            cov = derive_covariates(raw, y.index, base['enso_lag_days'])
        except Exception as exc:
            print(f'GAGAL mengambil kovariat iklim nyata: {exc}')
            sys.exit(2)

    wanted = {m.strip() for m in a.model.split(',') if m.strip()}
    sweeps = [s for s in SWEEPS if not wanted or s[0] in wanted]
    cache, rows = {}, []
    csv_path = os.path.join(a.output, 'sensitivitas_parameter.csv')
    done = set()
    if a.lanjut and os.path.exists(csv_path):
        old_rows = pd.read_csv(csv_path, dtype={'Nilai': str})
        if not cek_bawaan(old_rows, 'hasil lama'):
            print('\nHasil lama tidak konsisten dengan Tabel 4.2 (kemungkinan dijalankan di lingkungan Python lain).\n'
                  'Hapus folder keluaran lalu jalankan ulang tanpa --lanjut di virtual environment yang sama dengan Bab IV.')
            sys.exit(3)
        rows = old_rows.to_dict('records')
        done = {(r['Model'], r['Parameter'], str(r['Nilai'])) for r in rows}
        print(f'Melanjutkan: {len(rows)} baris sudah ada, hanya baris yang belum ada yang dihitung.', flush=True)

    def run(model, key, value, windows=test_w, train_end=split):
        cfg = merge_config(QUICK if a.cepat else {})
        use_cov = cov
        if key is None:
            pass  # konfigurasi bawaan (dipakai sebagai model penyusun Smart Ensemble)
        elif key == 'kovariat':
            use_cov = cov if value else None
        elif model == 'Naive Seasonal':
            cfg['seasonal_period'] = value
        elif model == 'SMA':
            cfg['sma_window'] = value
        elif model == 'ARIMA':
            cfg['arima_order'] = value
        elif model == 'Prophet':
            cfg['prophet'][key] = value
        elif model == 'BiLSTM':
            cfg['lstm'][key] = value
        elif model == 'TFT':
            cfg['tft'][key] = value
        ck = (model, json.dumps(cfg, sort_keys=True, default=str), use_cov is not None, windows is val_w)
        if ck in cache:
            return cache[ck]
        t0 = time.time()
        if model == 'Naive Seasonal':
            p = run_naive_seasonal(y, windows, cfg['seasonal_period'])
        elif model == 'SMA':
            p = run_sma(y, windows, cfg['sma_window'])
        elif model == 'ARIMA':
            p = run_arima(y, windows, tuple(cfg['arima_order']))
        elif model == 'Prophet':
            p = run_prophet(y, windows, use_cov, cfg['prophet'])
        elif model == 'BiLSTM':
            p = run_bilstm(y, windows, train_end, cfg['lstm'], cfg['seed'], with_mc=False)
        else:
            p = run_tft(y, windows, train_end, a.provinsi, a.komoditas, use_cov, H, cfg['tft'], cfg['seed'])
        cache[ck] = (p, time.time() - t0)
        return cache[ck]

    for model, label, key, values, default in sweeps:
        if a.cepat and key == 'max_epochs':
            values = [1, 2]
        for v in values:
            if (model, label, fmt_val(v)) in done:
                continue
            print(f'{model:15s} {label} = {fmt_val(v)}', flush=True)
            if model == 'Smart Ensemble':
                from models.ensemble import SmartEnsemble
                t0 = time.time()
                val_p = {m: run(m, None, None, val_w, val_start)[0]['pred'] for m in ['Prophet', 'BiLSTM', 'TFT']}
                test_p = {m: run(m, None, None)[0] for m in ['Prophet', 'BiLSTM', 'TFT']}
                ens = SmartEnsemble()
                info = ens.fit_weights_grid(val_p, y, step=v if key == 'grid_step' else 0.05)
                fb = v if key == 'fallback_cv' else 0.15
                comb, flag = ens.combine_series({m: test_p[m]['pred'] for m in test_p}, fallback_cv=fb)
                base_df = test_p['BiLSTM'][['origin_date', 'horizon', 'y_origin']].copy()
                base_df['pred'] = comb.reindex(base_df.index).values
                pred, secs = base_df, time.time() - t0
                extra = ', '.join(f'{k} {w * 100:.0f}%' for k, w in info['weights'].items())
            else:
                pred, secs = run(model, key, v)
                extra = ''
            m = _model_metrics(model, pred, y)
            rows.append({'Model': model, 'Parameter': label, 'Nilai': fmt_val(v), 'Bawaan': v == default,
                         'MAPE (%)': m['MAPE (%)'], 'RMSE': m['RMSE'], 'MAE': m['MAE'],
                         'Directional Accuracy (%)': m['Directional Accuracy (%)'],
                         'Dalam ±10% (%)': m['Dalam ±10% (%)'], 'Waktu (detik)': round(secs, 1), 'Catatan': extra})
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    res = pd.DataFrame(rows)
    order = {(m, l, fmt_val(v)): i for i, (m, l, k, vals, d) in enumerate(SWEEPS) for v in vals}
    res['_urut'] = [order.get((r['Model'], r['Parameter'], str(r['Nilai'])), 999) for _, r in res.iterrows()]
    res = res.sort_values('_urut', kind='stable').drop(columns='_urut').reset_index(drop=True)
    res.to_csv(csv_path, index=False)
    print('\nCek konsistensi baris bawaan terhadap Tabel 4.2:')
    cek_bawaan(res, 'hasil akhir')
    # Grafik: MAPE terhadap nilai parameter, satu panel per parameter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 7})
    groups = list(res.groupby(['Model', 'Parameter'], sort=False))
    cols = 3
    rws = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(6.6, 1.75 * rws), squeeze=False)
    for ax, ((model, param), g) in zip(axes.flat, groups):
        colors = ['#c44e52' if b else '#4c72b0' for b in g['Bawaan']]
        ax.bar(range(len(g)), g['MAPE (%)'], color=colors)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g['Nilai'], fontsize=6, rotation=0)
        ax.set_title(f'{model}: {param}', fontsize=6.8)
        ax.set_ylabel('MAPE (%)', fontsize=6)
        ax.grid(axis='y', color='#eeeeee')
        for i, val in enumerate(g['MAPE (%)']):
            ax.text(i, val, angka(val), ha='center', va='bottom', fontsize=5.6)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis('off')
    fig.suptitle('Batang merah = nilai bawaan', fontsize=7, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(a.output, 'gambar_sensitivitas_parameter.png'), dpi=220)
    plt.close(fig)

    md = ['# Ringkasan analisis sensitivitas parameter', '',
          f'{a.komoditas}, {a.provinsi}; uji {n - split} hari, {len(test_w)} titik asal; kovariat iklim: '
          f'{"ya" if cov is not None else "TIDAK"}.', '']
    if a.cepat or a.tanpa_kovariat:
        md.append('> PERINGATAN: mode uji alur; jangan dipakai di naskah.\n')
    md += ['| Model | Parameter | Nilai | MAPE (%) | Δ MAPE vs bawaan | RMSE | DA (%) | Waktu (detik) |',
           '|---|---|---|---|---|---|---|---|']
    for (model, param), g in groups:
        d = g.loc[g['Bawaan'], 'MAPE (%)']
        d = float(d.iloc[0]) if len(d) else np.nan
        for _, r in g.iterrows():
            md.append(f"| {model} | {param} | {r['Nilai']}{' (bawaan)' if r['Bawaan'] else ''} | "
                      f"{angka(r['MAPE (%)'])} | {angka(r['MAPE (%)'] - d)} | {angka(r['RMSE'], 0)} | "
                      f"{angka(r['Directional Accuracy (%)'], 1)} | {angka(r['Waktu (detik)'], 1)} |")
    open(os.path.join(a.output, 'ringkasan_parameter.md'), 'w').write('\n'.join(md) + '\n')
    print(f'Selesai. Keluaran: {a.output}')


if __name__ == '__main__':
    main()
