"""Tests for Data Sync Scheduler."""
import pytest
import pandas as pd
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


class TestDataSyncScheduler:
    def setup_method(self):
        from data.database import DataStore
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "test_scheduler.db"
        )
        self.store = DataStore(db_path=self.db_path, csv_path="nonexistent.csv")

    def teardown_method(self):
        self.store.close()
        for ext in ['', '-wal', '-shm']:
            path = self.db_path + ext
            if os.path.exists(path):
                os.remove(path)

    def test_run_once_no_existing_data(self):
        """Sync should log 'skipped' when no data exists."""
        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(self.store)
        count = scheduler.run_once()
        assert count == 0
        
        history = self.store.get_sync_history()
        assert len(history) == 1
        assert history[0]['status'] == 'skipped_no_data'

    def test_run_once_data_up_to_date(self):
        """Sync should log 'up_to_date' when last date is today."""
        today = datetime.now().strftime('%Y-%m-%d')
        self.store.insert_prices([
            {'date': today, 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000}
        ])
        
        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(self.store)
        count = scheduler.run_once()
        assert count == 0
        
        history = self.store.get_sync_history()
        assert history[0]['status'] == 'up_to_date'

    @patch('pihps_scraper.PihpsScraper')
    @patch('pihps_scraper.get_province_list')
    def test_run_once_fetches_new_data(self, mock_get_provs, mock_scraper_cls):
        """Sync should fetch and insert new data when behind."""
        # Set last_date to 5 days ago
        five_days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        self.store.insert_prices([
            {'date': five_days_ago, 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000}
        ])
        
        # Mock scraper
        mock_scraper = MagicMock()
        mock_scraper.fetch_data.return_value = {"data": []}
        today_str = datetime.now().strftime('%Y-%m-%d')
        mock_scraper.process_json.return_value = [
            {'date': today_str, 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10500}
        ]
        mock_scraper_cls.return_value = mock_scraper
        mock_get_provs.return_value = [{'id': 13, 'name': 'DKI Jakarta'}]
        
        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(self.store)
        count = scheduler.run_once()
        
        assert count >= 1
        history = self.store.get_sync_history()
        assert history[0]['status'] == 'ok'

    def test_scheduler_init(self):
        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(self.store)
        assert scheduler.store is self.store
        assert scheduler._scheduler is None
