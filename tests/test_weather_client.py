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


def _fake_open_meteo(archive_available=True, null_days=()):
    """requests.get palsu: mengembalikan data harian sesuai rentang yang diminta."""
    from data.weather_client import WeatherClient as WC
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=7))).date()
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        if url == WC.ARCHIVE_URL:
            if not archive_available:
                raise RuntimeError("archive down")
            days = pd.date_range(params['start_date'], params['end_date'])
        else:
            days = pd.date_range(today - timedelta(days=params['past_days']),
                                 today + timedelta(days=params['forecast_days'] - 1))
        rain = [None if d.strftime('%Y-%m-%d') in null_days else 5.0 + (d.day % 7) for d in days]
        resp = MagicMock()
        resp.json.return_value = {'daily': {
            'time': [d.strftime('%Y-%m-%d') for d in days],
            'precipitation_sum': rain,
            'temperature_2m_mean': [26.0 + (d.day % 5) for d in days],
        }}
        return resp

    return fake_get, calls, today


class TestOpenMeteoCoverage:
    """Regresi: rentang panjang yang berakhir dekat hari ini tidak boleh diisi konstanta."""

    def test_long_range_ending_today_uses_archive_and_forecast(self):
        fake_get, calls, today = _fake_open_meteo()
        dates = pd.date_range(pd.Timestamp(today) - pd.Timedelta(days=800), pd.Timestamp(today))
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            client = WeatherClient(use_live=True, strict=True)
            rain = client.get_rainfall('Jawa Timur', dates)
        urls = [c[0] for c in calls]
        assert WeatherClient.ARCHIVE_URL in urls and WeatherClient.FORECAST_URL in urls
        forecast_params = [p for u, p in calls if u == WeatherClient.FORECAST_URL][0]
        assert forecast_params['past_days'] <= 92
        assert len(rain) == len(dates)
        assert rain.notna().all()
        # Tidak ada deretan panjang nilai 0 hasil pengisian
        assert (rain == 0).sum() == 0
        assert client.fallback_used == set()

    def test_missing_history_raises_in_strict_mode(self):
        fake_get, _, today = _fake_open_meteo(archive_available=False)
        dates = pd.date_range(pd.Timestamp(today) - pd.Timedelta(days=400), pd.Timestamp(today))
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            client = WeatherClient(use_live=True, strict=True)
            with pytest.raises(RuntimeError):
                client.get_rainfall('Jawa Timur', dates)
            assert client._fetch_open_meteo('Jawa Timur', dates) is None

    def test_missing_history_uses_flagged_fallback_when_not_strict(self):
        fake_get, _, today = _fake_open_meteo(archive_available=False)
        dates = pd.date_range(pd.Timestamp(today) - pd.Timedelta(days=400), pd.Timestamp(today))
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            client = WeatherClient(use_live=True, strict=False)
            client.get_temperature('Jawa Timur', dates)
        assert 'suhu:Jawa Timur' in client.fallback_used

    def test_short_gap_is_filled_long_gap_rejected(self):
        dates = pd.date_range('2024-01-01', '2024-01-31')
        short = {'2024-01-10', '2024-01-11'}
        fake_get, _, _ = _fake_open_meteo(null_days=short)
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            result = WeatherClient(use_live=True)._fetch_open_meteo('Bali', dates)
        assert result is not None and result.notna().all().all()

        long_gap = {d.strftime('%Y-%m-%d') for d in pd.date_range('2024-01-05', '2024-01-20')}
        fake_get, _, _ = _fake_open_meteo(null_days=long_gap)
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            assert WeatherClient(use_live=True)._fetch_open_meteo('Bali', dates) is None

    def test_dates_with_time_of_day_are_normalized(self):
        fake_get, _, today = _fake_open_meteo()
        now = pd.Timestamp.now()
        dates = pd.date_range(now - pd.Timedelta(days=7), now - pd.Timedelta(days=1))
        with patch('data.weather_client.requests.get', side_effect=fake_get):
            client = WeatherClient(use_live=True, strict=True)
            rain = client.get_rainfall('DKI Jakarta', dates)
        assert len(rain) == 7
        assert rain.notna().all()
        assert (rain.index == rain.index.normalize()).all()


class TestEnsoCache:
    def test_cache_used_when_covering_and_named_enso_index(self):
        client = WeatherClient(use_live=True)
        client._enso_cache = pd.Series([0.5, 0.7], index=pd.to_datetime(['2024-01-15', '2024-02-15']), name='oni')
        dates = pd.date_range('2024-02-01', '2024-02-20')
        with patch('data.weather_client.requests.get') as mock_get:
            out = client._fetch_noaa_enso(dates)
        mock_get.assert_not_called()
        assert out.name == 'enso_index'
        assert out.notna().all()

    def test_cache_not_covering_triggers_refetch(self):
        client = WeatherClient(use_live=True)
        client._enso_cache = pd.Series([0.5], index=pd.to_datetime(['2024-01-15']), name='oni')
        dates = pd.date_range('2023-12-01', '2023-12-10')
        resp = MagicMock()
        resp.text = "SEAS  YR   TOTAL   ANOM\nNDJ 2023 28.0 1.9\nDJF 2024 27.9 1.8\n"
        with patch('data.weather_client.requests.get', return_value=resp) as mock_get:
            out = client._fetch_noaa_enso(dates)
        mock_get.assert_called_once()
        assert out.name == 'enso_index'
