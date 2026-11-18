"""
Helper jendela waktu berbasis tanggal kalender untuk data PIHPS.

PIHPS tidak terbit di akhir pekan/hari libur, jadi "7 hari" harus dihitung
dari tanggal, bukan jumlah baris (iloc[-7] bisa mencakup 9-10 hari).
"""
from typing import List, Optional, Tuple

import pandas as pd

# Seri yang tanggal terakhirnya tertinggal lebih dari ini dianggap tidak terkini
STALE_TOLERANCE_DAYS = 14


def price_days_ago(series: pd.DataFrame, days: int,
                   date_col: str = "date", value_col: str = "price") -> Optional[float]:
    """Nilai terakhir pada atau sebelum (tanggal terakhir - days). None jika tidak ada."""
    if series is None or series.empty:
        return None
    s = series.sort_values(date_col)
    dates = pd.to_datetime(s[date_col])
    cutoff = dates.iloc[-1] - pd.Timedelta(days=days)
    past = s[dates <= cutoff]
    if past.empty:
        return None
    return float(past[value_col].iloc[-1])


def pct_change_days(series: pd.DataFrame, days: int,
                    date_col: str = "date", value_col: str = "price") -> Optional[float]:
    """Perubahan % antara nilai terakhir dan nilai `days` hari kalender sebelumnya."""
    past = price_days_ago(series, days, date_col, value_col)
    if past is None or past <= 0:
        return None
    current = float(series.sort_values(date_col)[value_col].iloc[-1])
    return (current - past) / past * 100


def last_n_days(series: pd.DataFrame, days: int, date_col: str = "date") -> pd.DataFrame:
    """Baris dalam `days` hari kalender terakhir (relatif ke tanggal terakhir seri)."""
    if series is None or series.empty:
        return series
    s = series.sort_values(date_col)
    dates = pd.to_datetime(s[date_col])
    return s[dates >= dates.iloc[-1] - pd.Timedelta(days=days)]


def split_stale(latest: pd.DataFrame, tolerance_days: int = STALE_TOLERANCE_DAYS,
                date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pisahkan baris 'terbaru' menjadi (terkini, tertinggal).

    Baris tertinggal = tanggalnya lebih tua dari tanggal terakhir keseluruhan
    dikurangi `tolerance_days`, agar harga April tidak dicampur dengan November.
    """
    if latest is None or latest.empty:
        return latest, latest.iloc[0:0] if latest is not None else latest
    dates = pd.to_datetime(latest[date_col])
    cutoff = dates.max() - pd.Timedelta(days=tolerance_days)
    return latest[dates >= cutoff], latest[dates < cutoff]


def stale_caption(stale: pd.DataFrame, label_cols: List[str] = ("province",),
                  date_col: str = "date") -> str:
    """Teks ringkas daftar seri yang dikeluarkan karena datanya tidak terkini."""
    if stale is None or stale.empty:
        return ""
    parts = []
    for _, r in stale.sort_values(list(label_cols)).iterrows():
        name = " / ".join(str(r[c]) for c in label_cols)
        parts.append(f"{name} (terakhir {pd.Timestamp(r[date_col]).strftime('%d %b %Y')})")
    return ("Tidak ditampilkan karena datanya belum diperbarui lebih dari "
            f"{STALE_TOLERANCE_DAYS} hari: " + ", ".join(parts) + ".")


def balanced_national_series(df: pd.DataFrame, date_col: str = "date",
                             value_col: str = "price", group_col: str = "province",
                             fill_limit: int = 5) -> pd.DataFrame:
    """Rata-rata nasional harian dengan himpunan provinsi yang tetap.

    groupby(date).mean() memakai provinsi yang kebetulan melapor hari itu
    (29-34, berubah-ubah), sehingga muncul lonjakan semu. Di sini hanya
    provinsi yang punya data terkini yang dipakai, celah pendek diisi ffill,
    dan tanggal yang belum lengkap untuk semua provinsi itu dibuang.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[date_col, value_col])
    pivot = df.pivot_table(index=date_col, columns=group_col, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().ffill(limit=fill_limit)
    current_cols = pivot.columns[pivot.iloc[-1].notna()]
    balanced = pivot[current_cols].dropna(how="any")
    out = balanced.mean(axis=1).rename(value_col).reset_index()
    out.columns = [date_col, value_col]
    return out
