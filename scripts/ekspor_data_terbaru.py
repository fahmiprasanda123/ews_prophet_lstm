#!/usr/bin/env python3
"""
Ekspor data harga yang lebih baru dari food_prices_real.csv ke data/prices_latest.csv.

ews_data.db tidak ikut di-push, sehingga Streamlit Cloud membangun database dari CSV.
File ini menambal selisihnya tanpa mengubah data beku skripsi.

Contoh:
    python scripts/ekspor_data_terbaru.py
"""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.path.join(ROOT, 'ews_data.db')
CSV_BEKU = os.path.join(ROOT, 'food_prices_real.csv')
CSV_TERBARU = os.path.join(ROOT, 'data', 'prices_latest.csv')


def main():
    batas = pd.to_datetime(pd.read_csv(CSV_BEKU, usecols=['date'])['date'], format='mixed').max()
    batas = batas.strftime('%Y-%m-%d')

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql(
            "SELECT date, province, commodity, price FROM prices WHERE date > ? ORDER BY date, province, commodity",
            conn, params=[batas])

    df.to_csv(CSV_TERBARU, index=False)
    akhir = df['date'].max() if not df.empty else batas
    print(f"{len(df)} baris ({batas} s.d. {akhir}) ditulis ke {CSV_TERBARU}")


if __name__ == '__main__':
    main()
