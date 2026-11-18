"""
SQLite Data Layer for Agri-AI EWS.
Replaces raw CSV access with structured database operations.
Falls back to CSV if SQLite is unavailable.
"""
import math
import sqlite3
import pandas as pd
import os
import threading
from contextlib import contextmanager
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ews_data.db")
CSV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "food_prices_real.csv")

class DataStore:
    """Thread-safe SQLite data store with CSV fallback."""

    def __init__(self, db_path=None, csv_path=None):
        self.db_path = db_path or DB_FILE
        self.csv_path = csv_path or CSV_FILE
        self._conn = None
        # Koneksi dipakai bersama thread sesi Streamlit dan thread scheduler;
        # tanpa kunci, baca-tulis yang bersamaan memicu "database disk image is malformed".
        self._lock = threading.RLock()
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    @contextmanager
    def _get_conn(self):
        """Yield the persistent SQLite connection, held exclusively by one thread at a time."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            try:
                yield self._conn
            except Exception:
                self._conn.rollback()
                raise

    def close(self):
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    province TEXT NOT NULL,
                    commodity TEXT NOT NULL,
                    price REAL NOT NULL,
                    UNIQUE(date, province, commodity)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON prices(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prov ON prices(province)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comm ON prices(commodity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prov_comm ON prices(province, commodity)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_time TEXT NOT NULL,
                    records_added INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ok'
                )
            """)
            # Kolom pesan ditambahkan belakangan; DB lama perlu dimigrasi.
            cols = [r[1] for r in conn.execute("PRAGMA table_info(sync_log)").fetchall()]
            if 'message' not in cols:
                conn.execute("ALTER TABLE sync_log ADD COLUMN message TEXT")
            conn.commit()

    # ------------------------------------------------------------------ #
    # Migration
    # ------------------------------------------------------------------ #
    def migrate_from_csv(self, csv_path=None, progress_callback=None):
        """Import CSV data into SQLite. Skips if data already exists."""
        csv_path = csv_path or self.csv_path
        if not os.path.exists(csv_path):
            return 0

        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            if count > 0:
                return count  # Already migrated

        if progress_callback:
            progress_callback("Reading CSV file...")

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
        df = df.drop_duplicates(subset=['date', 'province', 'commodity'])

        if progress_callback:
            progress_callback(f"Importing {len(df)} records to database...")

        with self._get_conn() as conn:
            df.to_sql('prices', conn, if_exists='append', index=False,
                       method='multi', chunksize=5000)
            conn.commit()

        if progress_callback:
            progress_callback(f"Migration complete: {len(df)} records.")

        return len(df)

    # ------------------------------------------------------------------ #
    # Query operations
    # ------------------------------------------------------------------ #
    def load_all(self) -> pd.DataFrame:
        """Load all price data as a DataFrame."""
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]

        if count > 0:
            with self._get_conn() as conn:
                df = pd.read_sql("SELECT date, province, commodity, price FROM prices ORDER BY date", conn)
            df['date'] = pd.to_datetime(df['date'])
            return df
        elif os.path.exists(self.csv_path):
            # Fallback to CSV
            df = pd.read_csv(self.csv_path)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            return df
        else:
            return pd.DataFrame(columns=['date', 'province', 'commodity', 'price'])

    def get_provinces(self) -> list:
        """Get sorted list of unique provinces."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT province FROM prices ORDER BY province").fetchall()
        return [r[0] for r in rows]

    def get_commodities(self) -> list:
        """Get sorted list of unique commodities."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT commodity FROM prices ORDER BY commodity").fetchall()
        return [r[0] for r in rows]

    def get_series(self, province: str, commodity: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get price series for a specific province-commodity pair."""
        query = "SELECT date, price FROM prices WHERE province=? AND commodity=?"
        params = [province, commodity]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        with self._get_conn() as conn:
            df = pd.read_sql(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_latest_prices(self, commodity: str = None) -> pd.DataFrame:
        """Get the latest price for each province-commodity combo."""
        query = """
            SELECT p.date, p.province, p.commodity, p.price
            FROM prices p
            INNER JOIN (
                SELECT province, commodity, MAX(date) as max_date
                FROM prices
        """
        params = []
        if commodity:
            query += " WHERE commodity=?"
            params.append(commodity)

        query += """
                GROUP BY province, commodity
            ) latest ON p.province = latest.province
                    AND p.commodity = latest.commodity
                    AND p.date = latest.max_date
            ORDER BY p.province
        """

        with self._get_conn() as conn:
            df = pd.read_sql(query, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_last_date(self) -> str:
        """Get the most recent date in the dataset."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
        return row[0] if row and row[0] else None

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #
    def insert_prices(self, records: list, update_existing: bool = False) -> int:
        """Insert price records and return the number of rows actually inserted/changed.
        records: list of dicts with keys date, province, commodity, price

        Duplicates are ignored, unless update_existing=True: then a different price
        for an existing (date, province, commodity) replaces the stored one.
        Records with an empty, NaN, or non-positive price are skipped.
        """
        if not records:
            return 0

        rows = []
        for rec in records:
            row = _clean_record(rec)
            if row is not None:
                rows.append(row)
        if not rows:
            return 0

        if update_existing:
            sql = ("INSERT INTO prices (date, province, commodity, price) VALUES (?, ?, ?, ?) "
                   "ON CONFLICT(date, province, commodity) DO UPDATE SET price = excluded.price "
                   "WHERE prices.price IS NOT excluded.price")
        else:
            sql = "INSERT OR IGNORE INTO prices (date, province, commodity, price) VALUES (?, ?, ?, ?)"

        with self._get_conn() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            conn.commit()
            return conn.total_changes - before

    def log_sync(self, records_added: int, status: str = "ok", message: str = None):
        """Log a sync event."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sync_log (sync_time, records_added, status, message) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), int(records_added), status, message)
            )
            conn.commit()

    def get_sync_history(self, limit: int = 10) -> list:
        """Get recent sync log entries."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT sync_time, records_added, status, message FROM sync_log ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{"time": r[0], "records": r[1], "status": r[2], "message": r[3]} for r in rows]

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            provinces = conn.execute("SELECT COUNT(DISTINCT province) FROM prices").fetchone()[0]
            commodities = conn.execute("SELECT COUNT(DISTINCT commodity) FROM prices").fetchone()[0]
            date_range = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()

        return {
            "total_records": total,
            "provinces": provinces,
            "commodities": commodities,
            "date_from": date_range[0],
            "date_to": date_range[1],
        }


def _clean_record(rec):
    """Ubah satu record ke tipe bawaan Python; None bila harga tidak valid.

    Tipe numpy (mis. numpy.int64) tersimpan sebagai BLOB oleh sqlite3, jadi dikonversi dulu.
    """
    try:
        price = float(rec['price'])
        date = pd.Timestamp(rec['date']).strftime('%Y-%m-%d')
    except (TypeError, ValueError, KeyError):
        return None
    if math.isnan(price) or price <= 0:
        return None
    return (date, str(rec['province']), str(rec['commodity']), price)


# Convenience singleton
_store = None
_store_lock = threading.Lock()

def get_store() -> DataStore:
    """Get or create the global DataStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = DataStore()
    return _store
