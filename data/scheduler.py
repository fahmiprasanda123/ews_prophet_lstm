"""
Background scheduler for automatic PIHPS data synchronization.
Uses APScheduler to run daily sync tasks.
"""
import logging
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# WIB tidak memakai DST, jadi offset tetap +7 cukup (tanpa bergantung tzdata).
WIB = timezone(timedelta(hours=7), 'WIB')
# Hari terakhir selalu diambil ulang: data PIHPS bisa terlambat masuk atau dikoreksi.
LOOKBACK_DAYS = 7

# Satu sinkronisasi per proses; pemanggil lain langsung ditolak, bukan menunggu.
_sync_lock = threading.Lock()


class DataSyncScheduler:
    """Manages scheduled PIHPS data synchronization."""

    def __init__(self, data_store, scraper=None, province_list=None):
        self.store = data_store
        self.scraper = scraper
        self.province_list = province_list or []
        self._scheduler = None
        # Ringkasan sinkronisasi terakhir: status, records, message, failed.
        self.last_result = None

    def _do_sync(self):
        """Execute a single sync cycle."""
        from pihps_scraper import PihpsScraper, get_province_list

        scraper = self.scraper or PihpsScraper()
        provinces = self.province_list or get_province_list()

        last_date = self.store.get_last_date()
        if not last_date:
            logger.warning("No existing data found. Skipping incremental sync.")
            return self._finish(0, "skipped_no_data")

        today = datetime.now(WIB).date()
        last_dt = datetime.strptime(last_date, '%Y-%m-%d').date() if isinstance(last_date, str) else last_date
        start_dt = min(last_dt + timedelta(days=1), today - timedelta(days=LOOKBACK_DAYS))

        start_sync = start_dt.strftime('%Y-%m-%d')
        end_sync = today.strftime('%Y-%m-%d')
        logger.info(f"Syncing PIHPS data from {start_sync} to {end_sync}...")

        new_records = []
        failed = []
        for prov in provinces:
            try:
                raw = scraper.fetch_data(prov['id'], start_sync, end_sync)
                if raw is None:
                    # fetch_data mengembalikan None bila semua percobaan gagal
                    failed.append(prov['name'])
                    continue
                new_records.extend(scraper.process_json(raw, prov['name']))
            except Exception as e:
                logger.error(f"Failed to sync {prov['name']}: {e}")
                failed.append(prov['name'])

        changed = self.store.insert_prices(new_records, update_existing=True) if new_records else 0

        if provinces and len(failed) == len(provinces):
            status = "error"
            message = f"Semua {len(failed)} provinsi gagal diambil dari PIHPS."
        elif failed:
            status = "partial"
            message = f"{len(failed)} dari {len(provinces)} provinsi gagal: {', '.join(failed)}"
        else:
            status = "ok" if changed else "up_to_date"
            message = None
        logger.info(f"Sync {status}: {changed} records added/updated ({start_sync} to {end_sync}).")
        return self._finish(changed, status, message, failed)

    def _finish(self, records, status, message=None, failed=None):
        self.store.log_sync(records, status, message)
        self.last_result = {"status": status, "records": records,
                            "message": message, "failed": failed or []}
        return records

    def _safe_sync(self):
        """Jalankan _do_sync; error dicatat ke sync_log agar tidak hilang di thread APScheduler."""
        if not _sync_lock.acquire(blocking=False):
            logger.info("Sync already running; skipped.")
            self.last_result = {"status": "busy", "records": 0,
                                "message": "Sinkronisasi lain sedang berjalan.", "failed": []}
            return 0
        try:
            return self._do_sync()
        except Exception as e:
            logger.exception("Sync failed")
            try:
                return self._finish(0, "error", f"{type(e).__name__}: {e}")
            except Exception:
                logger.exception("Failed to log sync error")
                return 0
        finally:
            _sync_lock.release()

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
                self._safe_sync()
            return

        if self._scheduler is not None:
            logger.warning("Scheduler already running.")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._safe_sync,
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
        """Run a single sync cycle manually. Details are in self.last_result."""
        return self._safe_sync()
