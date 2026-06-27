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

