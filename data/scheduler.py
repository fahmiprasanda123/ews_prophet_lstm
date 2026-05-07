"""
Background scheduler for automatic PIHPS data synchronization.
Uses APScheduler to run daily sync tasks.
"""
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataSyncScheduler:
    """Manages scheduled PIHPS data synchronization."""

    def __init__(self, data_store, scraper=None, province_list=None):
        self.store = data_store
        self.scraper = scraper
        self.province_list = province_list or []
        self._scheduler = None

    def _do_sync(self):
        """Execute a single sync cycle."""
        from pihps_scraper import PihpsScraper, get_province_list

        scraper = self.scraper or PihpsScraper()
        provinces = self.province_list or get_province_list()

        last_date = self.store.get_last_date()
        if not last_date:
            logger.warning("No existing data found. Skipping incremental sync.")
            self.store.log_sync(0, "skipped_no_data")
            return 0

        today = datetime.now()
        last_dt = datetime.strptime(last_date, '%Y-%m-%d') if isinstance(last_date, str) else last_date
        days_behind = (today - last_dt).days

        if days_behind <= 1:
            logger.info("Data is already up to date.")
            self.store.log_sync(0, "up_to_date")
            return 0

        start_sync = (last_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        end_sync = today.strftime('%Y-%m-%d')
        logger.info(f"Syncing PIHPS data from {start_sync} to {end_sync}...")

        new_records = []
        for prov in provinces:
            try:
                raw = scraper.fetch_data(prov['id'], start_sync, end_sync)
                if raw:
                    tidy = scraper.process_json(raw, prov['name'])
                    new_records.extend(tidy)
            except Exception as e:
                logger.error(f"Failed to sync {prov['name']}: {e}")

        if new_records:
            inserted = self.store.insert_prices(new_records)
            self.store.log_sync(inserted, "ok")
            logger.info(f"Sync complete: {inserted} new records.")
            return inserted
        else:
            self.store.log_sync(0, "no_new_data")
            return 0

    def start(self, interval_hours=24, run_immediately=True):
        """Start the background scheduler.
        
        Args:
            interval_hours: Hours between sync cycles.
            run_immediately: If True, run one sync immediately before scheduling.
        """
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            logger.warning("APScheduler not installed. Scheduler disabled.")
            if run_immediately:
                self._do_sync()
            return

        if self._scheduler is not None:
            logger.warning("Scheduler already running.")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._do_sync,
            'interval',
            hours=interval_hours,
            id='pihps_sync',
            next_run_time=datetime.now() if run_immediately else None
        )
        self._scheduler.start()
        logger.info(f"Scheduler started (interval: {interval_hours}h)")

    def stop(self):
        """Stop the background scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("Scheduler stopped.")

    def run_once(self):
        """Run a single sync cycle manually."""
        return self._do_sync()
