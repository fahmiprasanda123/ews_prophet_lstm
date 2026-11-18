"""
Weather data integration module for Agri-AI EWS.
Uses Open-Meteo API (free, no API key) for real weather data.
Falls back to seasonal patterns when API is unavailable.

Data source: Open-Meteo (https://open-meteo.com) — CC BY 4.0 License
"""
import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Province capital coordinates for Open-Meteo API
PROVINCE_COORDS = {
    "Aceh":                         (-5.55, 95.32),
    "Sumatera Utara":               (3.59, 98.67),
    "Sumatera Barat":               (-0.95, 100.35),
    "Riau":                         (0.51, 101.45),
    "Kepulauan Riau":               (1.07, 104.03),
    "Jambi":                        (-1.61, 103.61),
    "Bengkulu":                     (-3.80, 102.26),
    "Sumatera Selatan":             (-2.99, 104.76),
    "Kepulauan Bangka Belitung":    (-2.13, 106.12),
    "Lampung":                      (-5.45, 105.26),
    "Banten":                       (-6.12, 106.15),
    "DKI Jakarta":                  (-6.20, 106.85),
    "Jawa Barat":                   (-6.91, 107.61),
    "Jawa Tengah":                  (-6.97, 110.42),
    "DI Yogyakarta":                (-7.80, 110.36),
    "Jawa Timur":                   (-7.25, 112.75),
    "Bali":                         (-8.65, 115.22),
    "Nusa Tenggara Barat":          (-8.58, 116.10),
    "Nusa Tenggara Timur":          (-10.18, 123.60),
    "Kalimantan Barat":             (-0.03, 109.33),
    "Kalimantan Selatan":           (-3.32, 114.59),
    "Kalimantan Tengah":            (-2.21, 113.92),
    "Kalimantan Timur":             (-1.24, 116.83),
    "Kalimantan Utara":             (3.07, 116.04),
    "Sulawesi Utara":               (1.49, 124.84),
    "Gorontalo":                    (0.56, 123.06),
    "Sulawesi Selatan":             (-5.14, 119.42),
    "Sulawesi Tenggara":            (-3.97, 122.51),
    "Sulawesi Tengah":              (-0.90, 121.49),
    "Sulawesi Barat":               (-2.84, 119.23),
    "Maluku":                       (-3.69, 128.18),
    "Maluku Utara":                 (1.72, 127.81),
    "Papua":                        (-2.54, 140.72),
    "Papua Barat":                  (-0.86, 134.06),
}

# Hari terakhir yang belum tentu ada di archive API; bagian ini diambil dari forecast API.
_ARCHIVE_LAG_DAYS = 5
# Batas past_days pada forecast API Open-Meteo.
_FORECAST_MAX_PAST_DAYS = 92
_FORECAST_MAX_DAYS = 16
# Celah data hilang terpanjang (hari) yang masih boleh diisi nilai tetangga.
_MAX_GAP_DAYS = 3


def _normalize_dates(dates) -> pd.DatetimeIndex:
    """Buang jam dari tanggal (mis. dari Timestamp.now()) agar cocok dengan data harian."""
    return pd.DatetimeIndex(dates).normalize()


# Seasonal fallback (for when API fails)
_SEASONAL_RAINFALL = {
    1: 300, 2: 280, 3: 250, 4: 200, 5: 120,
    6: 80,  7: 60,  8: 50,  9: 70,  10: 130,
    11: 220, 12: 290
}


class WeatherClient:
    """Provides real weather data from Open-Meteo API for Indonesian provinces.
    
    Open-Meteo is a free, open-source weather API that provides:
    - Historical data from 1940 to present
    - Daily precipitation, temperature, humidity, wind
    - No API key required
    - CC BY 4.0 license (attribution required)
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, use_live=True, cache_days=7, strict=False):
        """
        Args:
            use_live: If True, fetch from Open-Meteo API. If False, use fallback.
            cache_days: Number of days to cache API responses.
            strict: Jika True, kegagalan mengambil data nyata menimbulkan error
                alih-alih diam-diam memakai pola sintetis (pola musiman + angka
                acak). WAJIB True untuk evaluasi dan analisis tesis.
        """
        self.use_live = use_live
        self.cache_days = cache_days
        self.strict = strict
        self.fallback_used = set()
        self._cache = {}
        self._enso_cache = None

    def _fallback(self, kind, province=None):
        """Catat (atau tolak, bila strict) pemakaian data sintetis."""
        label = f"{kind}:{province}" if province else kind
        if self.strict:
            raise RuntimeError(
                f"Data {kind} nyata tidak tersedia ({province or 'nasional'}). "
                "Mode strict menolak data sintetis; periksa koneksi ke Open-Meteo/NOAA."
            )
        logger.warning(f"PERINGATAN: memakai data {kind} SINTETIS untuk {province or 'nasional'}")
        self.fallback_used.add(label)

    def get_rainfall(self, province: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Get daily precipitation data for a province.
        
        Returns a Series indexed by date with rainfall in mm.
        """
        dates = _normalize_dates(dates)
        if self.use_live:
            try:
                weather = self._fetch_open_meteo(province, dates)
                if weather is not None and 'precipitation_sum' in weather.columns:
                    return weather['precipitation_sum'].rename('rainfall_mm')
            except Exception as e:
                logger.warning(f"Open-Meteo fetch failed for {province}, using fallback: {e}")

        self._fallback('curah hujan', province)
        return self._generate_seasonal_rainfall(province, dates)

    def get_temperature(self, province: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Get daily mean temperature for a province.
        
        Returns a Series indexed by date with temperature in °C.
        """
        dates = _normalize_dates(dates)
        if self.use_live:
            try:
                weather = self._fetch_open_meteo(province, dates)
                if weather is not None and 'temperature_2m_mean' in weather.columns:
                    return weather['temperature_2m_mean'].rename('temperature_c')
            except Exception as e:
                logger.warning(f"Open-Meteo temp fetch failed: {e}")

        # Fallback: tropical average
        self._fallback('suhu', province)
        return pd.Series(
            np.random.normal(27.5, 1.5, len(dates)),
            index=dates, name='temperature_c'
        )

    def get_enso_index(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Get ENSO (El Niño/La Niña) ONI index from NOAA (public data).
        
        Positive = El Niño (drought risk), Negative = La Niña (flood risk).
        """
        dates = _normalize_dates(dates)
        try:
            return self._fetch_noaa_enso(dates)
        except Exception as e:
            logger.warning(f"NOAA ENSO fetch failed, using proxy: {e}")
            self._fallback('ENSO')
            return self._generate_enso_proxy(dates)

    def get_weather_features(self, province: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Get all weather features as a DataFrame for model input.
        
        Columns: rainfall_mm, temperature_c, enso_index, is_wet_season
        """
        dates = _normalize_dates(dates)
        rainfall = self.get_rainfall(province, dates)
        temperature = self.get_temperature(province, dates)
        enso = self.get_enso_index(dates)

        is_wet = pd.Series(
            [1 if m in [11, 12, 1, 2, 3, 4] else 0 for m in dates.month],
            index=dates
        )

        df = pd.DataFrame({
            'rainfall_mm': rainfall,
            'temperature_c': temperature,
            'enso_index': enso,
            'is_wet_season': is_wet,
        }, index=dates)

        return df

    # ------------------------------------------------------------------ #
    # Open-Meteo API
    # ------------------------------------------------------------------ #
    def _fetch_open_meteo(self, province: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Fetch historical weather data from Open-Meteo API.

        Free, no API key required, covers Indonesia from 1940-present.
        Tanggal lama diambil dari archive API, beberapa hari terakhir (dan masa
        depan) dari forecast API. Mengembalikan None bila data nyata tidak menutup
        seluruh tanggal yang diminta (celah > _MAX_GAP_DAYS hari), agar pemanggil
        memakai jalur fallback/strict alih-alih nilai konstan buatan.
        """
        coords = PROVINCE_COORDS.get(province)
        if coords is None:
            logger.warning(f"No coordinates for province: {province}")
            return None

        lat, lon = coords
        dates = _normalize_dates(dates)

        # Check cache
        cache_key = f"{province}_{dates.min().date()}_{dates.max().date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_dt = dates.min().date()
        end_dt = dates.max().date()
        today = datetime.now(timezone(timedelta(hours=7))).date()
        archive_end = today - timedelta(days=_ARCHIVE_LAG_DAYS + 1)
        base = {
            'latitude': lat,
            'longitude': lon,
            'daily': 'precipitation_sum,temperature_2m_mean',
            'timezone': 'Asia/Jakarta',
        }

        try:
            parts = []
            if start_dt <= archive_end:
                parts.append(self._request_daily(self.ARCHIVE_URL, dict(
                    base,
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=min(end_dt, archive_end).strftime('%Y-%m-%d'),
                )))
            if end_dt > archive_end:
                recent_start = max(start_dt, archive_end + timedelta(days=1))
                parts.append(self._request_daily(self.FORECAST_URL, dict(
                    base,
                    past_days=min(_FORECAST_MAX_PAST_DAYS, max(0, (today - recent_start).days)),
                    forecast_days=min(_FORECAST_MAX_DAYS, max(1, (end_dt - today).days + 1)),
                )))

            result = pd.concat(parts)
            result = result[~result.index.duplicated(keep='first')].sort_index()
            result = result.reindex(dates)

            # Hanya celah pendek yang diisi; celah panjang berarti data nyata tidak tersedia.
            na = result.isna().any(axis=1)
            longest_gap = int(na.groupby((~na).cumsum()).sum().max()) if na.any() else 0
            if longest_gap > _MAX_GAP_DAYS:
                logger.warning(
                    f"Open-Meteo data for {province} does not cover {start_dt}..{end_dt} "
                    f"({int(na.sum())} of {len(dates)} days missing, longest gap {longest_gap})"
                )
                return None
            result = result.interpolate(limit_area='inside').ffill().bfill()

            # Cache the result
            self._cache[cache_key] = result
            logger.info(f"✅ Open-Meteo: fetched {len(result)} days for {province} ({lat}, {lon})")

            return result

        except requests.exceptions.Timeout:
            logger.warning(f"Open-Meteo timeout for {province}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Open-Meteo HTTP error for {province}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Open-Meteo error for {province}: {e}")
            return None

    def _request_daily(self, url: str, params: dict) -> pd.DataFrame:
        """Satu permintaan Open-Meteo -> DataFrame harian; nilai null dibiarkan NaN."""
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if 'daily' not in data:
            raise ValueError("No daily data in Open-Meteo response")
        daily = data['daily']
        n = len(daily['time'])
        result = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'precipitation_sum': pd.to_numeric(pd.Series(daily.get('precipitation_sum', [None] * n)), errors='coerce'),
            'temperature_2m_mean': pd.to_numeric(pd.Series(daily.get('temperature_2m_mean', [None] * n)), errors='coerce'),
        })
        return result.set_index('date')

    # ------------------------------------------------------------------ #
    # NOAA ENSO (public, no API key)
    # ------------------------------------------------------------------ #
    def _fetch_noaa_enso(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Fetch ENSO ONI data from NOAA (publicly available)."""
        if self._enso_cache is not None:
            # Pakai cache hanya bila menutup semua tanggal; bila tidak, unduh ulang.
            cached = self._enso_cache.reindex(dates, method='ffill')
            if not cached.isna().any():
                return cached.rename('enso_index')

        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        lines = response.text.strip().split('\n')
        records = []
        month_map = {
            'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
            'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
            'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
        }

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # NOAA format: SEAS YR TOTAL ANOM
                    season = parts[0].strip()
                    year = int(parts[1])
                    oni_value = float(parts[3])
                    if season in month_map:
                        records.append({
                            'year': year,
                            'month': month_map[season],
                            'oni': oni_value
                        })
                except (ValueError, IndexError):
                    continue

        if not records:
            raise ValueError("No ENSO data parsed from NOAA")

        enso_df = pd.DataFrame(records)
        enso_df['date'] = pd.to_datetime(
            enso_df['year'].astype(str) + '-' + enso_df['month'].astype(str) + '-15'
        )
        enso_series = enso_df.set_index('date')['oni'].sort_index()
        
        # Cache globally
        self._enso_cache = enso_series

        # Reindex to daily
        enso_daily = enso_series.reindex(dates, method='ffill').fillna(0)
        logger.info(f"✅ NOAA ENSO: fetched {len(records)} monthly records")
        return enso_daily.rename('enso_index')

    # ------------------------------------------------------------------ #
    # Fallback generators
    # ------------------------------------------------------------------ #
    def _generate_seasonal_rainfall(self, province: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate seasonal rainfall patterns as fallback."""
        factor = {
            "Kalimantan Barat": 1.3, "Papua": 1.3, "Sumatera Barat": 1.2,
            "Kalimantan Tengah": 1.2, "DKI Jakarta": 0.8, "Jawa Timur": 0.75,
            "Nusa Tenggara Timur": 0.5,
        }.get(province, 1.0)

        values = []
        for dt in dates:
            base = _SEASONAL_RAINFALL.get(dt.month, 150)
            noise = np.random.normal(1.0, 0.2)
            daily_mm = (base / 30) * factor * max(0.1, noise)
            values.append(round(daily_mm, 1))

        return pd.Series(values, index=dates, name='rainfall_mm')

    def _generate_enso_proxy(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Generate ENSO-like signal as fallback."""
        days_from_ref = (dates - pd.Timestamp('2020-01-01')).days
        enso = 0.8 * np.sin(2 * np.pi * days_from_ref / (365.25 * 3.7))
        enso += np.random.normal(0, 0.15, len(dates))
        return pd.Series(np.round(enso, 2), index=dates, name='enso_index')

    def get_data_source_info(self) -> dict:
        """Return info about the data source for attribution."""
        return {
            'weather': {
                'provider': 'Open-Meteo',
                'url': 'https://open-meteo.com',
                'license': 'CC BY 4.0',
                'type': 'live' if self.use_live else 'fallback',
            },
            'enso': {
                'provider': 'NOAA Climate Prediction Center',
                'url': 'https://www.cpc.ncep.noaa.gov',
                'license': 'Public Domain',
            }
        }
