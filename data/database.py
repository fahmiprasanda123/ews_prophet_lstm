"""
SQLite Data Layer for Agri-AI EWS.
Replaces raw CSV access with structured database operations.
Falls back to CSV if SQLite is unavailable.
"""
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
        self._init_db()

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    @contextmanager
    def _get_conn(self):
        """Yield a persistent SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def close(self):
        """Close the database connection."""
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
    def insert_prices(self, records: list) -> int:
        """Insert new price records. Skips duplicates.
        records: list of dicts with keys date, province, commodity, price
        """
        if not records:
            return 0

        inserted = 0
        with self._get_conn() as conn:
            for rec in records:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO prices (date, province, commodity, price) VALUES (?, ?, ?, ?)",
                        (rec['date'], rec['province'], rec['commodity'], rec['price'])
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()

        return inserted

    def log_sync(self, records_added: int, status: str = "ok"):
        """Log a sync event."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sync_log (sync_time, records_added, status) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), records_added, status)
            )
            conn.commit()

    def get_sync_history(self, limit: int = 10) -> list:
        """Get recent sync log entries."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT sync_time, records_added, status FROM sync_log ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{"time": r[0], "records": r[1], "status": r[2]} for r in rows]

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


# Convenience singleton
_store = None

def get_store() -> DataStore:
    """Get or create the global DataStore singleton."""
    global _store
    if _store is None:
        _store = DataStore()
    return _store
