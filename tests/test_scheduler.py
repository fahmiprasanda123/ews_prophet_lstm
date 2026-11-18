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
        """Sync should log 'up_to_date' when re-fetched data changes nothing."""
        today = datetime.now().strftime('%Y-%m-%d')
        rec = {'date': today, 'province': 'DKI Jakarta', 'commodity': 'Beras', 'price': 10000}
        self.store.insert_prices([rec])

        scraper = MagicMock()
        scraper.fetch_data.return_value = {"data": []}
        scraper.process_json.return_value = [dict(rec)]

        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(self.store, scraper=scraper,
                                      province_list=[{'id': 13, 'name': 'DKI Jakarta'}])
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


class TestSyncWindowAndFailures:
    """Regresi: hari yang terambil sebagian harus diambil ulang; kegagalan harus tercatat."""

    PROVS = [{'id': 1, 'name': 'Aceh'}, {'id': 21, 'name': 'Kepulauan Riau'}]

    def setup_method(self):
        from data.database import DataStore
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_scheduler.db")
        self.store = DataStore(db_path=self.db_path, csv_path="nonexistent.csv")

    def teardown_method(self):
        self.store.close()
        for ext in ['', '-wal', '-shm']:
            if os.path.exists(self.db_path + ext):
                os.remove(self.db_path + ext)

    def _scheduler(self, scraper):
        from data.scheduler import DataSyncScheduler
        return DataSyncScheduler(self.store, scraper=scraper, province_list=self.PROVS)

    def _today(self):
        from data.scheduler import WIB
        return datetime.now(WIB).date()

    def test_partial_day_is_refetched_and_corrected(self):
        today = self._today()
        d2 = (today - timedelta(days=2)).strftime('%Y-%m-%d')
        today_s = today.strftime('%Y-%m-%d')
        # Hari ini sudah ada (global MAX = hari ini), tetapi Kepri hilang 2 hari lalu
        # dan harga Aceh 2 hari lalu kemudian dikoreksi PIHPS.
        self.store.insert_prices([
            {'date': today_s, 'province': 'Aceh', 'commodity': 'Beras', 'price': 13000},
            {'date': d2, 'province': 'Aceh', 'commodity': 'Beras', 'price': 12000},
        ])
        scraper = MagicMock()
        scraper.fetch_data.return_value = {"data": ["x"]}
        scraper.process_json.side_effect = lambda raw, name: [
            {'date': d2, 'province': name, 'commodity': 'Beras', 'price': 12500},
            {'date': today_s, 'province': name, 'commodity': 'Beras', 'price': 13000},
        ]

        count = self._scheduler(scraper).run_once()

        start_arg = scraper.fetch_data.call_args[0][1]
        assert start_arg <= (today - timedelta(days=7)).strftime('%Y-%m-%d')
        # Kepri d2 + Kepri hari ini baru, Aceh d2 dikoreksi; Aceh hari ini tidak berubah
        assert count == 3
        kepri = self.store.get_series('Kepulauan Riau', 'Beras')
        assert len(kepri) == 2
        aceh = self.store.get_series('Aceh', 'Beras')
        assert aceh['price'].iloc[0] == 12500
        assert self.store.get_sync_history()[0]['status'] == 'ok'

    def test_all_provinces_fail_logs_error(self):
        self.store.insert_prices([{'date': '2025-01-01', 'province': 'Aceh', 'commodity': 'Beras', 'price': 1}])
        scraper = MagicMock()
        scraper.fetch_data.return_value = None
        sched = self._scheduler(scraper)
        assert sched.run_once() == 0
        hist = self.store.get_sync_history()[0]
        assert hist['status'] == 'error'
        assert hist['message']
        assert sched.last_result['failed'] == ['Aceh', 'Kepulauan Riau']

    def test_some_provinces_fail_logs_partial(self):
        self.store.insert_prices([{'date': '2025-01-01', 'province': 'Aceh', 'commodity': 'Beras', 'price': 1}])
        scraper = MagicMock()
        scraper.fetch_data.side_effect = [{"data": []}, RuntimeError("boom")]
        scraper.process_json.return_value = []
        sched = self._scheduler(scraper)
        sched.run_once()
        hist = self.store.get_sync_history()[0]
        assert hist['status'] == 'partial'
        assert 'Kepulauan Riau' in hist['message']

    def test_unexpected_exception_is_logged_not_raised(self):
        self.store.insert_prices([{'date': '2025-01-01', 'province': 'Aceh', 'commodity': 'Beras', 'price': 1}])
        sched = self._scheduler(MagicMock())
        with patch.object(self.store, 'insert_prices', side_effect=RuntimeError("disk")):
            sched.scraper.fetch_data.return_value = {"data": []}
            sched.scraper.process_json.return_value = [
                {'date': '2025-01-02', 'province': 'Aceh', 'commodity': 'Beras', 'price': 1}]
            assert sched.run_once() == 0
        hist = self.store.get_sync_history()[0]
        assert hist['status'] == 'error'
        assert 'disk' in hist['message']

    def test_concurrent_sync_is_rejected(self):
        from data import scheduler as sched_mod
        sched = self._scheduler(MagicMock())
        assert sched_mod._sync_lock.acquire(blocking=False)
        try:
            assert sched.run_once() == 0
            assert sched.last_result['status'] == 'busy'
            sched.scraper.fetch_data.assert_not_called()
        finally:
            sched_mod._sync_lock.release()
