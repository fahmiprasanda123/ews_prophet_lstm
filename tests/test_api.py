import pytest
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        
        # Seed test data
        from data.database import DataStore
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_api.db")
        store = DataStore(db_path=db_path)
        
        records = []
        from datetime import datetime, timedelta
        import numpy as np
        price = 10000
        for i in range(60):
            price += np.random.normal(0, 50)
            for prov in ['DKI Jakarta', 'Jawa Barat']:
                for comm in ['Beras', 'Cabai Merah']:
                    records.append({
                        'date': (datetime(2025, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'),
                        'province': prov,
                        'commodity': comm,
                        'price': round(price + np.random.normal(0, 500), 2),
                    })
        store.insert_prices(records)

        # Patch the global store
        import data.database as db_module
        db_module._store = store

        yield TestClient(app)

        # Cleanup
        store.close()
        db_module._store = None
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
            for ext in ['-wal', '-shm']:
                path = db_path + ext
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
    except ImportError:
        pytest.skip("FastAPI test dependencies not available")


class TestHealthEndpoints:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'online'

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200


class TestDataEndpoints:
    def test_get_commodities(self, client):
        response = client.get("/api/data/commodities")
        assert response.status_code == 200
        data = response.json()
        assert 'commodities' in data
        assert len(data['commodities']) > 0

    def test_get_provinces(self, client):
        response = client.get("/api/data/provinces")
        assert response.status_code == 200
        data = response.json()
        assert 'provinces' in data

    def test_get_prices(self, client):
        response = client.get("/api/data/prices?province=DKI+Jakarta&commodity=Beras&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_get_stats(self, client):
        response = client.get("/api/data/stats")
        assert response.status_code == 200
        data = response.json()
        assert data['total_records'] > 0

    def test_supply_risk(self, client):
        response = client.get("/api/data/supply-risk?province=DKI+Jakarta&commodity=Beras")
        assert response.status_code == 200
        data = response.json()
        assert 'score' in data
        assert 0 <= data['score'] <= 100

    def test_get_latest(self, client):
        response = client.get("/api/data/latest")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_latest_filtered(self, client):
        response = client.get("/api/data/latest?commodity=Beras")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for record in data:
            assert record['commodity'] == 'Beras'

    @patch('data.scheduler.DataSyncScheduler.run_once')
    def test_sync_endpoint(self, mock_run_once, client):
        mock_run_once.return_value = 5
        response = client.post("/api/data/sync")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert data['records_added'] == 5

    def test_ews_status(self, client):
        response = client.get("/api/ews/status")
        assert response.status_code == 200
        data = response.json()
        assert 'alerts' in data
        assert 'timestamp' in data
        assert 'total_danger' in data

    def test_ews_status_filtered(self, client):
        response = client.get("/api/ews/status?commodity=Beras")
        assert response.status_code == 200
        data = response.json()
        assert 'alerts' in data


class TestForecastEndpoints:
    def test_forecast_prophet(self, client):
        response = client.get("/api/forecast?province=DKI+Jakarta&commodity=Beras&days=7&model=prophet")
        assert response.status_code == 200
        data = response.json()
        assert data['province'] == 'DKI Jakarta'
        assert data['commodity'] == 'Beras'
        assert 'forecast' in data
        assert 'current_price' in data

    def test_forecast_invalid_model(self, client):
        response = client.get("/api/forecast?province=DKI+Jakarta&commodity=Beras&model=invalid")
        assert response.status_code in (400, 500)

    def test_forecast_missing_params(self, client):
        response = client.get("/api/forecast")
        assert response.status_code == 422  # Validation error

    def test_models_compare(self, client):
        response = client.get("/api/models/compare?province=DKI+Jakarta&commodity=Beras")
        assert response.status_code == 200
        data = response.json()
        assert 'models' in data
        assert 'best_model' in data



class TestRegressions:
    def test_prices_limit_must_be_positive(self, client):
        assert client.get("/api/data/prices?limit=-5").status_code == 422
        assert client.get("/api/data/prices?limit=0").status_code == 422

    def test_prices_invalid_date_rejected(self, client):
        assert client.get("/api/data/prices?start_date=bukan-tanggal").status_code == 422

    def test_prices_date_filter_consistent_between_paths(self, client):
        q = "start_date=2025-1-5&end_date=2025-1-10&limit=5000"
        sql_path = client.get(f"/api/data/prices?province=DKI+Jakarta&commodity=Beras&{q}").json()
        pd_path = client.get(f"/api/data/prices?province=DKI+Jakarta&{q}").json()
        pd_beras = [r for r in pd_path if r['commodity'] == 'Beras']
        assert len(sql_path) == 6
        assert [r['date'] for r in sql_path] == [r['date'] for r in pd_beras]
        assert sql_path[0]['date'] == '2025-01-05'

    def test_forecast_unknown_and_hybrid_model_return_400(self, client):
        for m in ("invalid", "hybrid"):
            r = client.get(f"/api/forecast?province=DKI+Jakarta&commodity=Beras&model={m}")
            assert r.status_code == 400, m

    def test_models_compare_unknown_series_404(self, client):
        r = client.get("/api/models/compare?province=Atlantis&commodity=Beras")
        assert r.status_code == 404

    def test_models_compare_all_fail_500_with_errors(self, client):
        with patch('models.prophet_forecast.FoodPriceProphet.prepare_data', side_effect=RuntimeError("p-fail")), \
             patch('models.lstm_forecast.LSTMForecaster.prepare_data', side_effect=RuntimeError("l-fail")):
            r = client.get("/api/models/compare?province=DKI+Jakarta&commodity=Beras")
        assert r.status_code == 500
        errors = r.json()['detail']['errors']
        assert 'p-fail' in errors['Prophet'] and 'l-fail' in errors['LSTM']

    def _sync_with_result(self, client, result, count=0):
        def fake_run_once(self):
            self.last_result = result
            return count
        with patch('data.scheduler.DataSyncScheduler.run_once', fake_run_once):
            return client.post("/api/data/sync")

    def test_sync_reports_error(self, client):
        r = self._sync_with_result(client, {'status': 'error', 'message': 'Semua 34 provinsi gagal', 'failed': []})
        assert r.status_code == 200
        assert r.json()['status'] == 'error'
        assert 'gagal' in r.json()['message']

    def test_sync_reports_partial(self, client):
        r = self._sync_with_result(client, {'status': 'partial', 'message': '1 dari 34 provinsi gagal', 'failed': []}, 10)
        assert r.json()['status'] == 'partial'
        assert r.json()['records_added'] == 10

    def test_sync_busy_returns_409(self, client):
        r = self._sync_with_result(client, {'status': 'busy', 'message': '', 'failed': []})
        assert r.status_code == 409


def test_ews_trend_uses_calendar_days():
    """Tren 7 hari dihitung dari harga terakhir pada/sebelum tanggal terakhir - 7 hari."""
    import pandas as pd
    from unittest.mock import MagicMock
    from api.routers import forecast as fc

    # Hanya hari kerja: iloc[-7] akan jatuh 9 hari kalender ke belakang
    dates = pd.bdate_range('2025-01-01', '2025-01-31')
    prices = [10000 + 100 * i for i in range(len(dates))]
    df = pd.DataFrame({'date': dates, 'province': 'Aceh', 'commodity': 'Beras', 'price': prices})
    store = MagicMock()
    store.load_all.return_value = df
    captured = {}

    class FakeEWS:
        def __init__(self, data):
            pass

        def calculate_composite_score(self, prov, comm, predicted):
            captured['predicted'] = predicted
            return {'level': 'Normal'}

    with patch.object(fc, 'get_store', return_value=store), \
         patch('engine.ews_engine_v2.EWSEngineV2', FakeEWS):
        fc.get_ews_status(commodity=None)

    last = df.iloc[-1]
    ref = df[df['date'] <= last['date'] - pd.Timedelta(days=7)]['price'].iloc[-1]
    assert captured['predicted'] == pytest.approx(last['price'] * (1 + (last['price'] - ref) / ref))
