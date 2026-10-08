#!/usr/bin/env python3
"""
Menghasilkan seluruh angka, tabel, dan gambar Bab IV tesis dari snapshot data beku.

Contoh pemakaian (jalankan dari folder utama repositori):
    python scripts/thesis_outputs.py                         # kombinasi utama + kovariat iklim (butuh internet)
    python scripts/thesis_outputs.py --lintas keduanya       # + validasi lintas komoditas & provinsi
    python scripts/thesis_outputs.py --uji-unit              # + ringkasan pengujian unit (pytest)
    python scripts/thesis_outputs.py --tanpa-kovariat --cepat   # uji alur cepat, BUKAN untuk naskah

Semua keluaran ditulis ke folder --output (bawaan: thesis_outputs/). Berkas
ringkasan_untuk_naskah.md memetakan setiap angka ke tabel/subbab di naskah.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.evaluation_protocol import (  # noqa: E402
    MODEL_ORDER, ALL_MODELS, climate_correlation, data_summary, evaluate_series,
    feature_importance, load_climate_covariates, load_frozen_data, predictions_frame,
    prepare_series, quiet_libraries,
)

QUICK = {'lstm': {'epochs': 3, 'mc_samples': 10}, 'tft': {'max_epochs': 1}}


# ---------------------------------------------------------------------- #
# Format angka gaya Indonesia
# ---------------------------------------------------------------------- #
def angka(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    s = f"{x:,.{d}f}"
    return s.replace(',', '_').replace('.', ',').replace('_', '.')


def md_table(df: pd.DataFrame, decimals: dict | None = None) -> str:
    decimals = decimals or {}
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(angka(float(v), decimals.get(c, 2)))
            elif isinstance(v, (int, np.integer)):
                cells.append(angka(int(v), 0))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------- #
# Gambar
# ---------------------------------------------------------------------- #
def plot_predictions(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    pf = predictions_frame(result)
    groups = [
        ('(a) Metode konvensional', ['Naive Seasonal', 'SMA-30', 'ARIMA(5,1,0)']),
        ('(b) Model AI dan Smart Ensemble', ['Prophet', 'BiLSTM', 'TFT', 'Smart Ensemble']),
    ]
    styles = {
        'Naive Seasonal': ('#9e9e9e', ':'), 'SMA-30': ('#616161', '--'), 'ARIMA(5,1,0)': ('#8d6e63', '-.'),
        'Prophet': ('#1f77b4', '--'), 'BiLSTM': ('#ff7f0e', '--'), 'TFT': ('#d62728', '-.'),
        'Smart Ensemble': ('#2ca02c', '-'),
    }
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8})
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.6), sharex=True)
    origins = sorted(set(pf['titik_asal']))
    for ax, (title, names) in zip(axes, groups):
        ax.plot(pf.index, pf['aktual'], color='black', lw=1.6, label='Aktual')
        for name in names:
            if name in pf.columns:
                color, ls = styles[name]
                ax.plot(pf.index, pf[name], color=color, ls=ls, lw=1.1, label=name)
        for o in origins:
            ax.axvline(o, color='#cccccc', lw=0.5, zorder=0)
        ax.set_title(title, loc='left', fontsize=8.5)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: angka(v / 1000, 0) + ' rb'))
        ax.set_ylabel('Harga (Rp/kg)')
        ax.legend(fontsize=7, ncol=4, loc='upper left', frameon=False)
        ax.grid(axis='y', color='#eeeeee')
    axes[-1].set_xlabel('Tanggal (garis tipis vertikal = titik asal prediksi)')
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_feature_importance(fi: pd.DataFrame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'Autolag harga': '#4c72b0', 'Lintas komoditas': '#dd8452', 'Temporal': '#55a868'}
    fi = fi.sort_values('|r|')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8})
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    bars = ax.barh(fi['Fitur'], fi['|r|'], color=[colors[g] for g in fi['Kelompok']])
    for bar, value in zip(bars, fi['|r|']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, angka(value, 3),
                va='center', fontsize=7)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Korelasi Pearson absolut |r| terhadap harga')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc='lower right', fontsize=7, frameon=False)
    ax.grid(axis='x', color='#eeeeee')
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# EWS per titik asal (tanpa hindsight)
# ---------------------------------------------------------------------- #
def ews_per_origin(df, result, horizon, rise_threshold=10.0):
    from engine.ews_engine_v2 import EWSEngineV2
    from engine.supply_risk import SupplyRiskScorer

    y = result['series']
    ens = result['predictions'].get('Smart Ensemble')
    if ens is None:
        return pd.DataFrame()
    rows = []
    for w in result['windows_test']:
        o, k, e = w['origin'], w['keep_from'], w['end']
        if e - k != horizon or e - o != horizon:
            continue
        asof, target = y.index[o - 1], y.index[e - 1]
        trunc = df[df['date'] <= asof]
        pred = float(ens.loc[target, 'pred'])
        ews = EWSEngineV2(trunc).calculate_composite_score(result['province'], result['commodity'], pred,
                                                           forecast_date=target)
        supply = SupplyRiskScorer(trunc).calculate_risk_score(result['province'], result['commodity'])
        current, actual = float(y.loc[asof]), float(y.loc[target])
        rows.append({
            'Tanggal data terakhir': asof.date().isoformat(),
            'Tanggal target': target.date().isoformat(),
            'Harga terakhir': current,
            'Prediksi ensemble': pred,
            'Harga aktual target': actual,
            'Perubahan prediksi (%)': (pred - current) / current * 100,
            'Perubahan aktual (%)': (actual - current) / current * 100,
            'Skor EWS': float(ews['score']),
            'Level EWS': ews['level'],
            'Skor risiko pasokan': float(supply['score']),
            f'Kenaikan aktual ≥ {rise_threshold:.0f}%': (actual - current) / current * 100 >= rise_threshold,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Pengujian unit
# ---------------------------------------------------------------------- #
def run_unit_tests(out_dir):
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(out_dir, 'pytest_hasil.xml')
    proc = subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=line', f'--junitxml={xml_path}'],
                          cwd=ROOT, capture_output=True, text=True)
    rows = {}
    if os.path.exists(xml_path):
        for case in ET.parse(xml_path).getroot().iter('testcase'):
            module = case.get('classname', '').split('.')[1] if '.' in case.get('classname', '') else case.get('classname')
            r = rows.setdefault(module, {'Modul uji': module, 'Lulus': 0, 'Gagal': 0, 'Dilewati': 0})
            if case.find('failure') is not None or case.find('error') is not None:
                r['Gagal'] += 1
            elif case.find('skipped') is not None:
                r['Dilewati'] += 1
            else:
                r['Lulus'] += 1
    table = pd.DataFrame(sorted(rows.values(), key=lambda r: r['Modul uji']))
    return table, proc.stdout[-2000:]


# ---------------------------------------------------------------------- #
# Utama
# ---------------------------------------------------------------------- #
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def versions():
    out = {'python': platform.python_version()}
    for mod in ['numpy', 'pandas', 'statsmodels', 'prophet', 'torch', 'pytorch_forecasting', 'lightning']:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = 'tidak terpasang'
    return out


def lintas_combos(df, mode, province, commodity):
    combos = []
    if mode in ('jatim', 'keduanya'):
        combos += [(province, c) for c in sorted(df['commodity'].unique())]
    if mode in ('bawangmerah', 'keduanya'):
        combos += [(p, commodity) for p in sorted(df['province'].unique())]
    if mode == 'semua':
        combos = [tuple(x) for x in df[['province', 'commodity']].drop_duplicates().values]
    seen, unique = set(), []
    for c in combos:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(ROOT, 'food_prices_real.csv'))
    ap.add_argument('--batas-tanggal', default=None, help='Tanggal akhir data beku (YYYY-MM-DD).')
    ap.add_argument('--provinsi', default='Jawa Timur')
    ap.add_argument('--komoditas', default='Bawang Merah')
    ap.add_argument('--output', default=os.path.join(ROOT, 'thesis_outputs'))
    ap.add_argument('--tanpa-kovariat', action='store_true', help='Jalankan tanpa kovariat iklim (uji alur).')
    ap.add_argument('--cepat', action='store_true', help='Epoch dikecilkan untuk uji alur, BUKAN untuk naskah.')
    ap.add_argument('--lintas', default='tidak', choices=['tidak', 'jatim', 'bawangmerah', 'keduanya', 'semua'])
    ap.add_argument('--lintas-model', default='naive,sma,arima,prophet,bilstm,ensemble',
                    help='Model untuk validasi lintas kombinasi (TFT lambat; tambahkan ",tft" bila perlu).')
    ap.add_argument('--uji-unit', action='store_true', help='Jalankan pytest dan simpan ringkasannya.')
    args = ap.parse_args()

    quiet_libraries()
    os.makedirs(args.output, exist_ok=True)
    t_start = time.time()
    config = QUICK if args.cepat else {}
    md = [f"# Ringkasan keluaran untuk naskah\n\nDibuat: {datetime.now():%d-%m-%Y %H:%M}  ",
          f"Data: `{os.path.basename(args.data)}` (SHA-256 `{sha256(args.data)[:16]}…`)  "]
    if args.cepat or args.tanpa_kovariat:
        md.append("\n> **PERINGATAN:** dijalankan dengan "
                  + ", ".join(x for x, f in [('--cepat', args.cepat), ('--tanpa-kovariat', args.tanpa_kovariat)] if f)
                  + ". Hasil ini hanya untuk uji alur dan TIDAK boleh dipakai di naskah.\n")

    df = load_frozen_data(args.data, args.batas_tanggal)
    print(f"Data: {len(df):,} catatan, {df['date'].min().date()} s.d. {df['date'].max().date()}")

    # 4.1 Ringkasan data
    total, per_series = data_summary(df)
    per_series.to_csv(os.path.join(args.output, '4_1_cakupan_per_deret.csv'), index=False)
    pd.Series(total).to_csv(os.path.join(args.output, '4_1_ringkasan_data.csv'), header=['Nilai'])
    md.append("\n## Subbab 4.1 — Tabel 4.1 Ringkasan pemeriksaan data\n")
    md.append(md_table(pd.DataFrame({'Butir': list(total.keys()), 'Nilai': [str(v) for v in total.values()]})))

    # Kovariat iklim
    cov_raw = None
    if not args.tanpa_kovariat:
        y_main, _ = prepare_series(df, args.provinsi, args.komoditas)
        try:
            cov_raw = load_climate_covariates(args.provinsi, y_main.index.min(), y_main.index.max(),
                                              cache_dir=os.path.join(ROOT, 'data', 'cache'), strict=True)
        except Exception as exc:
            print(f"\nGAGAL mengambil kovariat iklim nyata: {exc}\n"
                  "Periksa koneksi internet ke archive-api.open-meteo.com dan www.cpc.ncep.noaa.gov,\n"
                  "atau jalankan dengan --tanpa-kovariat HANYA untuk uji alur.")
            sys.exit(2)

    # 4.2 Evaluasi utama
    print(f"\nEvaluasi utama: {args.komoditas}, {args.provinsi}")
    result = evaluate_series(df, args.provinsi, args.komoditas, config=config, covariates_raw=cov_raw,
                             progress=lambda m, f: print(f"  [{f:4.0%}] {m}", flush=True))
    metrics = result['metrics']
    metrics.to_csv(os.path.join(args.output, '4_2_metrik_model.csv'), index=False)
    result['horizon_mape'].to_csv(os.path.join(args.output, '4_2_mape_per_horizon.csv'), index=False)
    result['intervals'].to_csv(os.path.join(args.output, '4_2_cakupan_interval.csv'), index=False)
    predictions_frame(result).to_csv(os.path.join(args.output, '4_2_prediksi_vs_aktual.csv'))
    plot_predictions(result, os.path.join(args.output, 'gambar_4_1_prediksi_vs_aktual.png'))
    with open(os.path.join(args.output, '4_2_bobot_ensemble.json'), 'w') as f:
        json.dump(result['ensemble'], f, indent=2, ensure_ascii=False)

    sp = result['split']
    md.append(f"\n## Subbab 4.2 — protokol\n\nData latih {sp['n_train']} hari (s.d. {sp['train_end']}), "
              f"data uji {sp['n_test']} hari ({sp['test_start']} s.d. {sp['test_end']}), "
              f"{sp['n_windows_test']} titik asal, horizon {result['config']['horizon']} hari. "
              f"Validasi bobot ensemble: {sp['n_val']} hari mulai {sp['val_start']}. "
              f"Kovariat iklim: {'ya' if result['covariates_used'] else 'TIDAK'}.\n")
    md.append("\n### Tabel 4.2 Perbandingan kinerja\n")
    cols = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)', 'R²', 'Directional Accuracy (%)', 'Kategori MAPE']
    md.append(md_table(metrics[cols], {'RMSE': 0, 'MAE': 0, 'R²': 3, 'Directional Accuracy (%)': 1}))
    md.append("\n### Persentase hari dengan galat dalam toleransi (untuk paragraf ±X%)\n")
    md.append(md_table(metrics[['Model', 'Dalam ±5% (%)', 'Dalam ±10% (%)']], {}))
    md.append("\n### Tabel 4.3 MAPE menurut horizon\n")
    md.append(md_table(result['horizon_mape']))
    if result['ensemble']:
        e = result['ensemble']
        md.append("\n### Bobot Smart Ensemble (hasil grid search pada validasi)\n")
        md.append(', '.join(f"{k} {angka(v * 100, 0)}%" for k, v in e['weights'].items())
                  + f"; MAPE validasi gabungan {angka(e['best_score'])}%; fallback aktif pada "
                  f"{e['fallback_days']} hari ({angka(e['fallback_rate'], 1)}% hari uji); "
                  f"model validasi terbaik: {e['best_single']}.")
    if not result['intervals'].empty:
        md.append("\n### Tabel 4.4 Cakupan interval ketidakpastian\n")
        md.append(md_table(result['intervals']))
    if result['notes']:
        md.append("\n**Catatan:** " + '; '.join(result['notes']))

    # 4.4 EWS
    ews = ews_per_origin(df, result, result['config']['horizon'])
    ews.to_csv(os.path.join(args.output, '4_4_skor_ews_per_titik_asal.csv'), index=False)
    if not ews.empty:
        md.append("\n## Subbab 4.4 — Tabel 4.6 Skor EWS per titik asal\n")
        md.append(md_table(ews, {'Harga terakhir': 0, 'Prediksi ensemble': 0, 'Harga aktual target': 0,
                                 'Perubahan prediksi (%)': 1, 'Perubahan aktual (%)': 1, 'Skor EWS': 1,
                                 'Skor risiko pasokan': 1}))

    # 4.5 Kepentingan fitur
    fi = feature_importance(df, args.provinsi, args.komoditas)
    fi.to_csv(os.path.join(args.output, '4_5_kepentingan_fitur.csv'), index=False)
    plot_feature_importance(fi, os.path.join(args.output, 'gambar_4_2_kepentingan_fitur.png'))
    md.append("\n## Subbab 4.5 — nilai korelasi tiap fitur (Gambar 4.2)\n")
    md.append(md_table(fi, {'|r|': 3}))

    # 4.6 Kovariat iklim
    if cov_raw is not None:
        cc = climate_correlation(result['series'], cov_raw)
        cc.to_csv(os.path.join(args.output, '4_6_korelasi_iklim.csv'), index=False)
        md.append("\n## Subbab 4.6 — Tabel 4.7 Korelasi kovariat iklim dan harga\n")
        md.append(md_table(cc, {'Pearson r': 3, 'p (Pearson)': 4, 'Spearman ρ': 3, 'p (Spearman)': 4}))

    # 4.3 Lintas kombinasi
    if args.lintas != 'tidak':
        combos = lintas_combos(df, args.lintas, args.provinsi, args.komoditas)
        models = [m.strip() for m in args.lintas_model.split(',') if m.strip() in ALL_MODELS]
        rows = []
        print(f"\nValidasi lintas: {len(combos)} kombinasi, model {models}")
        for i, (prov, comm) in enumerate(combos, 1):
            print(f"  [{i}/{len(combos)}] {comm}, {prov}", flush=True)
            try:
                cov_c = None
                if not args.tanpa_kovariat:
                    y_c, _ = prepare_series(df, prov, comm)
                    cov_c = load_climate_covariates(prov, y_c.index.min(), y_c.index.max(),
                                                    cache_dir=os.path.join(ROOT, 'data', 'cache'), strict=True)
                r = evaluate_series(df, prov, comm, config=config, models=models, covariates_raw=cov_c)
                for _, m in r['metrics'].iterrows():
                    rows.append({'Provinsi': prov, 'Komoditas': comm, 'Model': m['Model'],
                                 'MAPE (%)': m['MAPE (%)'], 'RMSE': m['RMSE'],
                                 'Directional Accuracy (%)': m['Directional Accuracy (%)']})
            except Exception as exc:
                print(f"    gagal: {exc}")
        lintas = pd.DataFrame(rows)
        lintas.to_csv(os.path.join(args.output, '4_3_lintas_kombinasi.csv'), index=False)
        if not lintas.empty:
            best = lintas.loc[lintas.groupby(['Provinsi', 'Komoditas'])['MAPE (%)'].idxmin()]
            summary = lintas.groupby('Model')['MAPE (%)'].agg(['mean', 'median', 'min', 'max']).reindex(
                [m for m in MODEL_ORDER if m in set(lintas['Model'])]).reset_index()
            summary['Jumlah kombinasi terbaik'] = summary['Model'].map(best['Model'].value_counts()).fillna(0).astype(int)
            summary.columns = ['Model', 'MAPE rata-rata (%)', 'MAPE median (%)', 'MAPE min (%)', 'MAPE maks (%)',
                               'Jumlah kombinasi terbaik']
            summary.to_csv(os.path.join(args.output, '4_3_ringkasan_lintas.csv'), index=False)
            md.append(f"\n## Subbab 4.3 — Tabel 4.5 validasi lintas ({len(combos)} kombinasi)\n")
            md.append(md_table(summary))

    # 4.8 Pengujian unit
    if args.uji_unit:
        table, tail = run_unit_tests(args.output)
        table.to_csv(os.path.join(args.output, '4_8_pengujian_unit.csv'), index=False)
        md.append("\n## Subbab 4.8 — Tabel 4.8 Hasil pengujian unit\n")
        md.append(md_table(table) if not table.empty else '```\n' + tail + '\n```')

    meta = {'dibuat': datetime.now().isoformat(timespec='seconds'), 'data': os.path.basename(args.data),
            'sha256_data': sha256(args.data), 'batas_tanggal': args.batas_tanggal,
            'provinsi': args.provinsi, 'komoditas': args.komoditas, 'kovariat_iklim': result['covariates_used'],
            'mode_cepat': args.cepat, 'konfigurasi': result['config'], 'pembagian': result['split'],
            'waktu_per_model_detik': result['timing'], 'versi_pustaka': versions(),
            'durasi_total_menit': round((time.time() - t_start) / 60, 1)}
    with open(os.path.join(args.output, 'konfigurasi.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(args.output, 'ringkasan_untuk_naskah.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f"\nSelesai dalam {meta['durasi_total_menit']} menit. Keluaran: {args.output}")


if __name__ == '__main__':
    main()
