#!/usr/bin/env python3
"""
Prakiraan harga ke depan dari data terbaru (untuk Sub-bab 4.9 naskah).

Contoh:
    python scripts/prakiraan_ke_depan.py                     # data database sistem (terbaru), 30 hari ke depan
    python scripts/prakiraan_ke_depan.py --sumber csv        # data beku food_prices_real.csv
    python scripts/prakiraan_ke_depan.py --tanpa-kovariat --cepat   # uji alur saja

Bobot Smart Ensemble dibaca dari thesis_outputs/4_2_bobot_ensemble.json (hasil evaluasi),
sehingga prakiraan konsisten dengan model yang dievaluasi di Bab IV.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.evaluation_protocol import load_climate_covariates, load_frozen_data, prepare_series  # noqa: E402
from models.forecast_ahead import _tgl, aggregate_forecast, forecast_path  # noqa: E402

QUICK = {'lstm': {'epochs': 3, 'mc_samples': 10}, 'tft': {'max_epochs': 1}}
MODELS = ['Naive Seasonal', 'SMA-30', 'ARIMA(5,1,0)', 'Prophet', 'BiLSTM', 'TFT', 'Smart Ensemble']


def angka(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def load_data(source, csv_path):
    if source == 'db':
        try:
            from data.database import get_store
            store = get_store()
            for name in ('load_all', 'get_all_prices', 'load_data'):
                if hasattr(store, name):
                    df = getattr(store, name)()
                    if df is not None and len(df):
                        df['date'] = pd.to_datetime(df['date'], format='mixed')
                        return df, 'database sistem'
        except Exception as exc:
            print(f'Database tidak dapat dibaca ({exc}); memakai CSV.')
    return load_frozen_data(csv_path), os.path.basename(csv_path)


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(angka(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sumber', choices=['db', 'csv'], default='db')
    ap.add_argument('--data', default=os.path.join(ROOT, 'food_prices_real.csv'))
    ap.add_argument('--provinsi', default='Jawa Timur')
    ap.add_argument('--komoditas', default='Bawang Merah')
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--bobot', default=os.path.join(ROOT, 'thesis_outputs', '4_2_bobot_ensemble.json'))
    ap.add_argument('--tanpa-kovariat', action='store_true')
    ap.add_argument('--cepat', action='store_true')
    ap.add_argument('--output', default=os.path.join(ROOT, 'prakiraan_outputs'))
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    df, sumber = load_data(a.sumber, a.data)
    y, _ = prepare_series(df, a.provinsi, a.komoditas)
    print(f'Data: {sumber}; {a.komoditas}, {a.provinsi}; data terakhir {y.index[-1].date()}')

    cov = None
    if not a.tanpa_kovariat:
        end = min(y.index[-1], pd.Timestamp.today().normalize() - pd.Timedelta(days=7))
        try:
            cov = load_climate_covariates(a.provinsi, y.index.min(), end,
                                          cache_dir=os.path.join(ROOT, 'data', 'cache'), strict=True)
        except Exception as exc:
            print(f'GAGAL mengambil kovariat iklim nyata: {exc}\nJalankan dengan --tanpa-kovariat hanya untuk uji alur.')
            sys.exit(2)

    weights, best = None, None
    if os.path.exists(a.bobot):
        w = json.load(open(a.bobot))
        weights, best = w['weights'], w.get('best_single')
    else:
        print('PERINGATAN: berkas bobot tidak ditemukan; Smart Ensemble tidak dihitung.')

    res = forecast_path(df, a.provinsi, a.komoditas, horizon=a.horizon, config=QUICK if a.cepat else {},
                        covariates_raw=cov, weights=weights, best_single=best,
                        progress=lambda m, f: print(f'  [{f:4.0%}] {m}', flush=True))
    path = res['path']
    show = [c for c in MODELS + ['Batas bawah', 'Batas atas'] if c in path.columns]
    daily = aggregate_forecast(path[show], 'D')
    weekly = aggregate_forecast(path[show], 'W')
    monthly = aggregate_forecast(path[show], 'M')
    daily.to_csv(os.path.join(a.output, 'prakiraan_harian.csv'), index=False)
    weekly.to_csv(os.path.join(a.output, 'prakiraan_mingguan.csv'), index=False)
    monthly.to_csv(os.path.join(a.output, 'prakiraan_bulanan.csv'), index=False)

    # Grafik: 90 hari terakhir + prakiraan
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8})
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    hist = y.iloc[-90:]
    ax.plot(hist.index, hist.values, color='black', lw=1.6, label='Harga aktual (90 hari terakhir)')
    if 'Batas bawah' in path:
        ax.fill_between(path.index, path['Batas bawah'], path['Batas atas'], color='#2ca02c', alpha=0.15,
                        label='Interval prakiraan')
    styles = {'Prophet': ('#1f77b4', '--'), 'BiLSTM': ('#ff7f0e', '--'), 'TFT': ('#d62728', '-.'),
              'ARIMA(5,1,0)': ('#8d6e63', ':')}
    for k, (c, ls) in styles.items():
        if k in path:
            ax.plot(path.index, path[k], color=c, ls=ls, lw=1.0, label=k)
    if 'Smart Ensemble' in path:
        ax.plot(path.index, path['Smart Ensemble'], color='#2ca02c', lw=2.0, label='Smart Ensemble')
    ax.axvline(y.index[-1], color='#999999', lw=0.8, ls='--')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: angka(v / 1000) + ' rb'))
    ax.set_ylabel('Harga (Rp/kg)')
    ax.legend(fontsize=6.5, ncol=3, frameon=False, loc='upper left')
    ax.grid(axis='y', color='#eeeeee')
    fig.tight_layout()
    fig.savefig(os.path.join(a.output, 'gambar_prakiraan_ke_depan.png'), dpi=220)
    plt.close(fig)

    meta = {'dibuat': datetime.now().isoformat(timespec='seconds'), 'sumber_data': sumber,
            'data_terakhir': str(y.index[-1].date()), 'harga_terakhir': res['last_price'],
            'horizon_hari': a.horizon, 'bobot': weights, 'kovariat_iklim': res['covariates_used'], 'mode_cepat': a.cepat}
    json.dump(meta, open(os.path.join(a.output, 'konfigurasi_prakiraan.json'), 'w'), indent=2, ensure_ascii=False)
    cols7 = ['Periode'] + [c for c in ['Smart Ensemble', 'Batas bawah', 'Batas atas', 'BiLSTM', 'Prophet', 'TFT', 'ARIMA(5,1,0)']
                           if c in daily.columns]
    md = [f'# Prakiraan ke depan — {a.komoditas}, {a.provinsi}', '',
          f'Dibuat {meta["dibuat"]}; sumber data: {sumber}; data terakhir {_tgl(y.index[-1])} '
          f'(Rp{angka(res["last_price"])}/kg); kovariat iklim: {"ya" if res["covariates_used"] else "TIDAK"}.', '',
          '## Tabel 4.9 — prakiraan harian 7 hari pertama (Rp/kg)', md_table(daily[cols7].head(7)), '',
          '## Tabel 4.10 — ringkasan mingguan (rata-rata, Rp/kg)', md_table(weekly[['Periode', 'Jumlah hari'] + cols7[1:]]), '',
          '## Ringkasan bulanan (rata-rata, Rp/kg)', md_table(monthly[['Periode', 'Jumlah hari'] + cols7[1:]]), '',
          'Gambar 4.4: gambar_prakiraan_ke_depan.png']
    if a.cepat or a.tanpa_kovariat:
        md.insert(2, '> PERINGATAN: mode uji alur (--cepat/--tanpa-kovariat); jangan dipakai di naskah.')
    open(os.path.join(a.output, 'ringkasan_prakiraan.md'), 'w').write('\n'.join(md) + '\n')
    print(daily[cols7].head(7).to_string(index=False))
    print(f'Keluaran: {a.output}')


if __name__ == '__main__':
    main()
