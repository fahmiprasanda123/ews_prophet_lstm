"""Tests for SQLite DataStore."""
import pytest
import pandas as pd
import os
import tempfile
from datetime import datetime


class TestDataStore:
    def setup_method(self):
        from data.database import DataStore
        # Use a temp file for testing
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "test_ews_data.db"
        )
        self.store = DataStore(db_path=self.db_path, csv_path="nonexistent.csv")

    def teardown_method(self):
        self.store.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        # Also remove WAL/SHM files
        for ext in ['-wal', '-shm']:
            path = self.db_path + ext
            if os.path.exists(path):
                os.remove(path)

    def test_init_creates_db(self):
        assert os.path.exists(self.db_path)

    def test_insert_and_query(self):
        records = [
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-02', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10100},
            {'date': '2025-01-01', 'province': 'Jawa Barat', 'commodity': 'Beras', 'price': 9500},
        ]
        inserted = self.store.insert_prices(records)
        assert inserted == 3

        df = self.store.load_all()
        assert len(df) == 3
        assert 'date' in df.columns
        assert 'price' in df.columns

    def test_get_provinces(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-01', 'province': 'Jawa Barat', 'commodity': 'Beras', 'price': 9500},
        ])
        provinces = self.store.get_provinces()
        assert 'DKI Jakarta' in provinces
        assert 'Jawa Barat' in provinces

    def test_get_commodities(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Cabai', 'price': 50000},
        ])
        commodities = self.store.get_commodities()
        assert 'Beras' in commodities
        assert 'Cabai' in commodities

    def test_get_series(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-02', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10100},
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Cabai', 'price': 50000},
        ])
        series = self.store.get_series('DKI Jakarta', 'Beras')
        assert len(series) == 2

    def test_get_latest_prices(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-02', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10100},
        ])
        latest = self.store.get_latest_prices()
        assert len(latest) == 1
        assert latest['price'].iloc[0] == 10100

    def test_duplicate_insert_ignored(self):
        rec = {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000}
        self.store.insert_prices([rec])
        self.store.insert_prices([rec])  # Duplicate
        
        df = self.store.load_all()
        assert len(df) == 1

    def test_get_stats(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-02', 'province': 'Jawa Barat', 'commodity': 'Cabai', 'price': 50000},
        ])
        stats = self.store.get_stats()
        assert stats['total_records'] == 2
        assert stats['provinces'] == 2
        assert stats['commodities'] == 2

    def test_sync_log(self):
        self.store.log_sync(100, "ok")
        history = self.store.get_sync_history()
        assert len(history) == 1
        assert history[0]['records'] == 100

    def test_empty_db(self):
        df = self.store.load_all()
        assert df.empty

    def test_get_last_date_empty(self):
        result = self.store.get_last_date()
        assert result is None

    def test_get_last_date_with_data(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-05', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10500},
        ])
        result = self.store.get_last_date()
        assert result == '2025-01-05'

    def test_migrate_from_csv_file_not_found(self):
        """Missing CSV should return 0 without error."""
        result = self.store.migrate_from_csv(csv_path="/nonexistent/path.csv")
        assert result == 0

    def test_migrate_from_csv_skips_if_data_exists(self):
        """Already-populated DB should skip migration."""
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'X', 'commodity': 'Y', 'price': 1000},
        ])
        result = self.store.migrate_from_csv(csv_path=self.db_path)
        assert result == 1  # Returns existing count, not re-importing

    def test_get_series_with_date_filter(self):
        self.store.insert_prices([
            {'date': '2025-01-01', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000},
            {'date': '2025-01-02', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10100},
            {'date': '2025-01-03', 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10200},
        ])
        series = self.store.get_series('DKI Jakarta', 'Beras', 
                                        start_date='2025-01-02', end_date='2025-01-03')
        assert len(series) == 2
        assert series['price'].iloc[0] == 10100

