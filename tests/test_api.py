"""Tests for FastAPI endpoints."""
import pytest
import os
import sys

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
        db_module._store = None
        if os.path.exists(db_path):
            os.remove(db_path)
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
