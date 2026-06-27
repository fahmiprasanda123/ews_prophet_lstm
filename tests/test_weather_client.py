"""Tests for Weather Client module."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from data.weather_client import WeatherClient, PROVINCE_COORDS, _SEASONAL_RAINFALL


class TestProvinceCoords:
    def test_has_34_provinces(self):
        assert len(PROVINCE_COORDS) == 34

    def test_all_coords_are_tuples(self):
        for prov, coords in PROVINCE_COORDS.items():
            assert isinstance(coords, tuple), f"{prov} coords is not a tuple"
            assert len(coords) == 2, f"{prov} coords should have 2 elements (lat, lon)"

    def test_coords_in_valid_range(self):
        """Indonesia spans roughly lat -11 to +6, lon 95 to 141."""
        for prov, (lat, lon) in PROVINCE_COORDS.items():
            assert -12 < lat < 7, f"{prov} latitude {lat} out of Indonesia range"
            assert 94 < lon < 142, f"{prov} longitude {lon} out of Indonesia range"

    def test_key_provinces_exist(self):
        for prov in ['DKI Jakarta', 'Jawa Barat', 'Jawa Timur', 'Aceh', 'Papua']:
            assert prov in PROVINCE_COORDS


class TestSeasonalRainfall:
    def test_12_months(self):
        assert len(_SEASONAL_RAINFALL) == 12

    def test_wet_season_higher(self):
        """Wet season months (Dec, Jan, Feb) should have higher rainfall."""
        wet_avg = np.mean([_SEASONAL_RAINFALL[m] for m in [12, 1, 2]])
        dry_avg = np.mean([_SEASONAL_RAINFALL[m] for m in [6, 7, 8]])
        assert wet_avg > dry_avg


class TestWeatherClient:
    def setup_method(self):
        # Use fallback mode (no API calls)
        self.client = WeatherClient(use_live=False)
        self.dates = pd.date_range('2024-06-01', '2024-06-10')

    def test_get_rainfall_fallback(self):
        rainfall = self.client.get_rainfall('DKI Jakarta', self.dates)
        assert len(rainfall) == len(self.dates)
        assert rainfall.name == 'rainfall_mm'
        assert all(rainfall >= 0)

    def test_get_temperature_fallback(self):
        temp = self.client.get_temperature('DKI Jakarta', self.dates)
        assert len(temp) == len(self.dates)
        assert temp.name == 'temperature_c'
        # Indonesia tropical temps
        assert temp.mean() > 20
        assert temp.mean() < 35

    def test_get_weather_features_shape(self):
        features = self.client.get_weather_features('DKI Jakarta', self.dates)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.dates)
        expected_cols = ['rainfall_mm', 'temperature_c', 'enso_index', 'is_wet_season']
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_is_wet_season_logic(self):
        """Wet season: Nov, Dec, Jan, Feb, Mar, Apr."""
        wet_dates = pd.date_range('2024-01-01', '2024-01-05')
        dry_dates = pd.date_range('2024-07-01', '2024-07-05')
        
        wet_features = self.client.get_weather_features('DKI Jakarta', wet_dates)
        dry_features = self.client.get_weather_features('DKI Jakarta', dry_dates)
        
        assert all(wet_features['is_wet_season'] == 1)
        assert all(dry_features['is_wet_season'] == 0)

    def test_get_data_source_info(self):
        info = self.client.get_data_source_info()
        assert 'weather' in info
        assert 'enso' in info
        assert info['weather']['provider'] == 'Open-Meteo'
        assert info['weather']['license'] == 'CC BY 4.0'
        assert info['enso']['provider'] == 'NOAA Climate Prediction Center'

    def test_fallback_mode_flag(self):
        client_live = WeatherClient(use_live=True)
        client_fallback = WeatherClient(use_live=False)
        
        info_live = client_live.get_data_source_info()
        info_fallback = client_fallback.get_data_source_info()
        
        assert info_live['weather']['type'] == 'live'
        assert info_fallback['weather']['type'] == 'fallback'

    def test_enso_proxy_fallback(self):
        """Proxy ENSO should return valid values in reasonable range."""
        enso = self.client._generate_enso_proxy(self.dates)
        assert len(enso) == len(self.dates)
        assert enso.name == 'enso_index'
        # ONI typically ranges -3 to +3
        assert enso.min() > -5
        assert enso.max() < 5

    def test_seasonal_rainfall_province_factor(self):
        """Provinces with higher factor should get more rainfall."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', '2024-01-30')
        
        wet_prov = self.client._generate_seasonal_rainfall('Kalimantan Barat', dates)
        dry_prov = self.client._generate_seasonal_rainfall('Nusa Tenggara Timur', dates)
        
        # Kalimantan Barat factor=1.3, NTT factor=0.5
        assert wet_prov.mean() > dry_prov.mean()

    @patch('data.weather_client.requests.get')
    def test_fetch_open_meteo_handles_timeout(self, mock_get):
        """API timeout should return None gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")
        
        client = WeatherClient(use_live=True)
        result = client._fetch_open_meteo('DKI Jakarta', self.dates)
        assert result is None

    @patch('data.weather_client.requests.get')
    def test_fetch_open_meteo_unknown_province(self, mock_get):
        """Province not in PROVINCE_COORDS should return None."""
        client = WeatherClient(use_live=True)
        result = client._fetch_open_meteo('Planet Mars', self.dates)
        assert result is None
        mock_get.assert_not_called()
