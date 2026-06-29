"""
Price Narrative Analyzer — Generates detailed explanations for price predictions.

Analyzes why a predicted price is going up, down, or staying stable by
examining seasonal patterns, weather effects, holiday impacts, historical
anomalies, recent trends, and regional factors specific to Indonesian food
commodities.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PriceNarrativeAnalyzer:
    """Generates human-readable narratives explaining price predictions."""

    # ------------------------------------------------------------------ #
    # Knowledge base: commodity-specific seasonal & sensitivity data
    # ------------------------------------------------------------------ #
    COMMODITY_KB = {
        # --- Beras (Rice) ---
        'Beras': {
            'harvest_months': [3, 4, 9, 10],  # Mar-Apr (rendeng), Sep-Oct (gadu)
            'scarce_months': [6, 7, 12, 1],
            'weather_sensitivity': 'high',
            'holiday_sensitivity': 'medium',
            'harvest_description': (
                'Musim panen raya padi (rendeng) biasanya terjadi pada Maret-April, '
                'dan panen gadu pada September-Oktober. Saat panen raya, pasokan '
                'beras melimpah sehingga harga cenderung turun.'
            ),
            'scarce_description': (
                'Periode Juni-Juli dan Desember-Januari merupakan masa paceklik '
                'beras dimana pasokan terbatas karena belum masuk musim panen, '
                'sehingga harga cenderung naik.'
            ),
            'weather_description': (
                'Beras sangat sensitif terhadap cuaca. Banjir dan kekeringan '
                'dapat merusak lahan sawah, menunda tanam, dan menurunkan '
                'produktivitas panen. Musim hujan berlebih (La Niña) berisiko '
                'banjir di area persawahan, sedangkan kemarau panjang (El Niño) '
                'menyebabkan kekeringan dan gagal panen.'
            ),
            'holiday_description': (
                'Menjelang Ramadan dan Idul Fitri, permintaan beras meningkat '
                'moderat untuk kebutuhan zakat fitrah dan konsumsi lebaran.'
            ),
        },
        # --- Cabai Merah ---
        'Cabai Merah': {
            'harvest_months': [1, 2, 3],
            'scarce_months': [7, 8, 9],
            'weather_sensitivity': 'very_high',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Musim panen cabai merah biasanya Januari-Maret. Saat panen '
                'raya, pasokan melimpah dan harga cenderung turun drastis, '
                'bahkan bisa anjlok hingga 50-70% dari harga puncak.'
            ),
            'scarce_description': (
                'Juli-September merupakan masa paceklik cabai karena musim '
                'kemarau mengurangi produktivitas. Pasokan berkurang signifikan '
                'dan harga bisa melonjak 2-3 kali lipat.'
            ),
            'weather_description': (
                'Cabai merah SANGAT sensitif terhadap cuaca. Hujan lebat '
                'menyebabkan penyakit busuk buah dan layu, mengurangi pasokan '
                'secara drastis. Kemarau panjang juga berbahaya karena tanaman '
                'cabai membutuhkan air yang cukup. Cuaca ekstrem adalah penyebab '
                'utama volatilitas harga cabai yang terkenal tinggi.'
            ),
            'holiday_description': (
                'Menjelang Ramadan dan Idul Fitri, permintaan cabai melonjak '
                'signifikan untuk kebutuhan masakan lebaran. Harga bisa naik '
                '30-100% dalam periode ini.'
            ),
        },
        # --- Cabai Rawit ---
        'Cabai Rawit': {
            'harvest_months': [1, 2, 3],
            'scarce_months': [7, 8, 9],
            'weather_sensitivity': 'very_high',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Panen cabai rawit umumnya Januari-Maret bersamaan dengan cabai '
                'merah. Pasokan melimpah menekan harga turun signifikan.'
            ),
            'scarce_description': (
                'Juli-September stok cabai rawit menipis. Harga cabai rawit '
                'terkenal sangat fluktuatif dan bisa melonjak sangat tinggi '
                'saat paceklik.'
            ),
            'weather_description': (
                'Cabai rawit sangat rentan terhadap hujan lebat yang menyebabkan '
                'busuk buah, serta kemarau panjang yang mengurangi hasil panen. '
                'Perubahan cuaca mendadak langsung berdampak pada pasokan.'
            ),
            'holiday_description': (
                'Permintaan cabai rawit naik tajam saat Ramadan dan Lebaran '
                'karena menjadi bumbu utama berbagai masakan tradisional.'
            ),
        },
        # --- Bawang Merah ---
        'Bawang Merah': {
            'harvest_months': [8, 9, 10],
            'scarce_months': [2, 3, 4],
            'weather_sensitivity': 'high',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Panen raya bawang merah terjadi Agustus-Oktober, terutama dari '
                'sentra produksi Brebes, Nganjuk, dan Enrekang. Pasokan melimpah '
                'menekan harga turun.'
            ),
            'scarce_description': (
                'Februari-April merupakan masa paceklik bawang merah. Stok gudang '
                'menipis dan petani belum panen, menyebabkan harga naik.'
            ),
            'weather_description': (
                'Bawang merah sensitif terhadap hujan berlebih yang menyebabkan '
                'penyakit jamur dan busuk umbi. Musim hujan panjang dapat '
                'mengurangi produksi secara signifikan. Kemarau moderat justru '
                'menguntungkan untuk budidaya bawang merah.'
            ),
            'holiday_description': (
                'Menjelang Ramadan, Idul Fitri, dan Natal, permintaan bawang '
                'merah meningkat drastis untuk kebutuhan masakan. Harga bisa '
                'naik 20-50%.'
            ),
        },
        # --- Bawang Putih ---
        'Bawang Putih': {
            'harvest_months': [7, 8, 9],
            'scarce_months': [1, 2, 3],
            'weather_sensitivity': 'medium',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Panen bawang putih domestik Juli-September, namun Indonesia '
                'masih bergantung pada impor (>90% dari China). Harga lebih '
                'dipengaruhi kebijakan impor dan ketersediaan stok impor.'
            ),
            'scarce_description': (
                'Januari-Maret sering terjadi kelangkaan bawang putih, terutama '
                'jika terjadi gangguan impor dari China (cuaca buruk atau '
                'kebijakan ekspor China).'
            ),
            'weather_description': (
                'Karena mayoritas impor, sensitivitas cuaca domestik relatif '
                'rendah. Namun, cuaca buruk di China atau gangguan logistik '
                'pengiriman laut dapat mempengaruhi pasokan dan harga.'
            ),
            'holiday_description': (
                'Permintaan bawang putih naik signifikan menjelang Ramadan dan '
                'Lebaran. Keterlambatan impor saat periode ini sangat kritis.'
            ),
        },
        # --- Daging Ayam ---
        'Daging Ayam': {
            'harvest_months': [],
            'scarce_months': [],
            'weather_sensitivity': 'low',
            'holiday_sensitivity': 'very_high',
            'harvest_description': (
                'Daging ayam tidak memiliki musim panen spesifik karena ternak '
                'dapat diproduksi sepanjang tahun. Harga lebih dipengaruhi oleh '
                'biaya pakan (jagung dan kedelai) dan permintaan pasar.'
            ),
            'scarce_description': (
                'Kelangkaan daging ayam jarang terjadi, namun bisa terjadi saat '
                'ada wabah penyakit unggas atau kenaikan drastis harga pakan.'
            ),
            'weather_description': (
                'Cuaca panas berlebih dapat meningkatkan angka kematian ayam dan '
                'mengurangi produksi. Secara umum, dampak cuaca relatif rendah '
                'dibanding komoditas pertanian.'
            ),
            'holiday_description': (
                'Harga daging ayam SANGAT sensitif terhadap hari raya. Menjelang '
                'Ramadan, Idul Fitri, Natal, dan Tahun Baru, permintaan melonjak '
                'drastis dan harga bisa naik 20-40%. Ini adalah faktor dominan '
                'pergerakan harga daging ayam.'
            ),
        },
        # --- Daging Sapi ---
        'Daging Sapi': {
            'harvest_months': [],
            'scarce_months': [],
            'weather_sensitivity': 'low',
            'holiday_sensitivity': 'very_high',
            'harvest_description': (
                'Daging sapi diproduksi sepanjang tahun. Indonesia masih impor '
                'daging sapi dan sapi bakalan dari Australia, India, dan Brazil. '
                'Harga sangat dipengaruhi kebijakan impor dan kurs Rupiah.'
            ),
            'scarce_description': (
                'Kelangkaan terjadi saat kuota impor terbatas atau ada gangguan '
                'pengiriman dari negara eksportir.'
            ),
            'weather_description': (
                'Dampak cuaca terhadap harga daging sapi relatif rendah. Kemarau '
                'panjang dapat mengurangi ketersediaan pakan ternak di beberapa '
                'daerah peternakan.'
            ),
            'holiday_description': (
                'Idul Adha adalah faktor TERBESAR untuk harga daging sapi. '
                'Permintaan melonjak luar biasa dan harga bisa naik 30-50%. '
                'Ramadan dan Idul Fitri juga mendorong kenaikan 15-30%. '
                'Natal dan Tahun Baru memiliki dampak moderat.'
            ),
        },
        # --- Telur Ayam ---
        'Telur Ayam': {
            'harvest_months': [],
            'scarce_months': [],
            'weather_sensitivity': 'medium',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Produksi telur berlangsung sepanjang tahun. Produktivitas ayam '
                'petelur dipengaruhi oleh umur ayam, kualitas pakan, dan kondisi '
                'kandang. Harga juga sangat dipengaruhi biaya pakan.'
            ),
            'scarce_description': (
                'Penurunan produksi bisa terjadi saat cuaca ekstrem atau ada '
                'wabah penyakit unggas yang mengurangi populasi ayam petelur.'
            ),
            'weather_description': (
                'Cuaca panas dapat menurunkan produktivitas ayam petelur '
                '(heat stress), mengurangi jumlah telur yang dihasilkan. '
                'Musim hujan juga bisa mempersulit distribusi.'
            ),
            'holiday_description': (
                'Permintaan telur meningkat signifikan menjelang Ramadan dan '
                'hari raya besar karena menjadi bahan utama kue dan makanan '
                'lebaran. Harga bisa naik 15-30%.'
            ),
        },
        # --- Minyak Goreng ---
        'Minyak Goreng': {
            'harvest_months': [3, 4, 5, 9, 10, 11],
            'scarce_months': [6, 7, 8, 12, 1, 2],
            'weather_sensitivity': 'low',
            'holiday_sensitivity': 'medium',
            'harvest_description': (
                'Panen kelapa sawit terjadi dua kali setahun (Maret-Mei dan '
                'September-November). Saat panen raya CPO, pasokan minyak goreng '
                'meningkat dan harga cenderung stabil atau turun.'
            ),
            'scarce_description': (
                'Di luar musim panen sawit, harga minyak goreng bisa naik '
                'terutama jika harga CPO global meningkat. Indonesia adalah '
                'eksportir CPO terbesar, sehingga harga global berpengaruh.'
            ),
            'weather_description': (
                'Dampak cuaca relatif rendah karena kelapa sawit cukup tahan '
                'terhadap variasi cuaca. Namun, kekeringan ekstrem bisa '
                'mengurangi produksi CPO dalam jangka panjang.'
            ),
            'holiday_description': (
                'Menjelang Ramadan dan Lebaran, konsumsi minyak goreng meningkat '
                'untuk kebutuhan menggoreng makanan lebaran. Harga bisa naik '
                '5-15%.'
            ),
        },
        # --- Gula Pasir ---
        'Gula Pasir': {
            'harvest_months': [5, 6, 7, 8, 9, 10, 11],
            'scarce_months': [12, 1, 2, 3, 4],
            'weather_sensitivity': 'low',
            'holiday_sensitivity': 'high',
            'harvest_description': (
                'Musim giling tebu (panen) berlangsung Mei-November. Saat musim '
                'giling, pasokan gula domestik meningkat dan harga cenderung stabil. '
                'Indonesia juga mengimpor gula rafinasi untuk industri.'
            ),
            'scarce_description': (
                'Desember-April merupakan masa di luar musim giling. Stok gula '
                'domestik bergantung pada sisa stok musim giling sebelumnya '
                'dan impor. Harga bisa naik jika stok menipis.'
            ),
            'weather_description': (
                'Kekeringan dapat mengurangi produktivitas tebu. Secara umum, '
                'dampak cuaca moderat karena gula juga bergantung pada impor.'
            ),
            'holiday_description': (
                'Permintaan gula melonjak menjelang Ramadan dan Lebaran untuk '
                'pembuatan kue dan minuman. Juga meningkat menjelang Natal dan '
                'Tahun Baru. Harga bisa naik 10-20%.'
            ),
        },
    }

    # Indonesian holiday calendar patterns (approximate)
    HOLIDAY_CALENDAR = {
        'ramadan': {
            'name': 'Ramadan & Idul Fitri',
            'pre_impact_days': 30,  # Price impact starts ~30 days before
            'post_impact_days': 7,
            'description': (
                'Bulan Ramadan dan menjelang Idul Fitri merupakan periode '
                'kenaikan permintaan bahan pokok paling signifikan di Indonesia. '
                'Konsumsi rumah tangga meningkat, belanja untuk hampers dan kue '
                'lebaran mendorong permintaan tinggi.'
            ),
        },
        'idul_adha': {
            'name': 'Idul Adha',
            'pre_impact_days': 14,
            'post_impact_days': 3,
            'description': (
                'Idul Adha (Hari Raya Kurban) mendorong kenaikan harga daging '
                'sapi dan kambing yang sangat signifikan. Permintaan hewan kurban '
                'melonjak drastis.'
            ),
        },
        'natal_tahun_baru': {
            'name': 'Natal & Tahun Baru',
            'pre_impact_days': 21,
            'post_impact_days': 7,
            'description': (
                'Periode Natal dan Tahun Baru mendorong peningkatan konsumsi '
                'bahan pokok terutama daging ayam, daging sapi, telur, dan '
                'bumbu masak.'
            ),
        },
    }

    # Weather seasons in Indonesia
    WEATHER_SEASONS = {
        'rainy': {
            'months': [10, 11, 12, 1, 2, 3],
            'name': 'Musim Hujan',
            'description': (
                'Musim hujan di Indonesia (Oktober-Maret) membawa curah hujan '
                'tinggi yang berdampak pada pertanian. Hujan lebat dapat '
                'menyebabkan banjir di lahan pertanian, gangguan distribusi, '
                'dan kerusakan tanaman. Namun, hujan juga diperlukan untuk '
                'pengairan sawah dan lahan tanam.'
            ),
        },
        'dry': {
            'months': [4, 5, 6, 7, 8, 9],
            'name': 'Musim Kemarau',
            'description': (
                'Musim kemarau (April-September) dapat menyebabkan kekeringan '
                'yang mengganggu produksi pertanian, terutama tanaman yang '
                'membutuhkan banyak air seperti padi. Namun, cuaca kering '
                'menguntungkan untuk komoditas tertentu seperti bawang merah.'
            ),
        },
    }

    def __init__(self, df):
        """
        Args:
            df: Full historical DataFrame with date, province, commodity, price.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def generate_narrative(self, province, commodity, predicted_price,
                           forecast_date=None):
        """Generate a complete narrative analysis explaining the price prediction.

        Args:
            province: Province name.
            commodity: Commodity name.
            predicted_price: Predicted future price.
            forecast_date: Target forecast date.

        Returns:
            dict with 'direction', 'summary', 'factors', 'narrative'
        """
        series = self.df[
            (self.df['province'] == province) &
            (self.df['commodity'] == commodity)
        ].sort_values('date')

        if series.empty or len(series) < 14:
            return self._fallback_narrative(predicted_price)

        current_price = series['price'].iloc[-1]

        if forecast_date is None:
            forecast_date = datetime.now()
        else:
            forecast_date = pd.Timestamp(forecast_date)

        # Determine direction
        pct_change = ((predicted_price - current_price) / current_price * 100
                      if current_price > 0 else 0)

        if pct_change > 2:
            direction = "NAIK"
        elif pct_change < -2:
            direction = "TURUN"
        else:
            direction = "STABIL"

        # Collect analysis factors
        factors = []

        # 1. Recent trend analysis
        trend_factor = self._analyze_recent_trend(series, pct_change)
        if trend_factor:
            factors.append(trend_factor)

        # 2. Seasonal / harvest analysis
        seasonal_factor = self._analyze_seasonal(
            series, commodity, forecast_date, pct_change
        )
        if seasonal_factor:
            factors.append(seasonal_factor)

        # 3. Weather season analysis
        weather_factor = self._analyze_weather_season(
            commodity, forecast_date, pct_change
        )
        if weather_factor:
            factors.append(weather_factor)

        # 4. Holiday effect analysis
        holiday_factor = self._analyze_holiday_effect(
            commodity, forecast_date, pct_change
        )
        if holiday_factor:
            factors.append(holiday_factor)

        # 5. Historical anomaly analysis
        anomaly_factor = self._analyze_historical_anomaly(
            series, predicted_price, forecast_date
        )
        if anomaly_factor:
            factors.append(anomaly_factor)

        # 6. Regional factor
        regional_factor = self._analyze_regional_factor(
            province, commodity, current_price, predicted_price
        )
        if regional_factor:
            factors.append(regional_factor)

        # Sort factors by impact
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        factors.sort(key=lambda x: impact_order.get(x['impact'], 3))

        # Build summary and narrative
        summary = self._build_summary(
            direction, pct_change, commodity, province, forecast_date, factors
        )
        narrative = self._build_narrative(
            direction, pct_change, commodity, province, forecast_date,
            current_price, predicted_price, factors
        )

        return {
            'direction': direction,
            'pct_change': round(pct_change, 1),
            'summary': summary,
            'factors': factors,
            'narrative': narrative,
        }

    # ------------------------------------------------------------------ #
    # Factor Analysis Methods
    # ------------------------------------------------------------------ #

    def _analyze_recent_trend(self, series, pct_change):
        """Analyze recent price movement trend."""
        if len(series) < 7:
            return None

        prices = series['price'].values

        # Calculate short-term trends
        changes = {}
        for label, window in [('7 hari', 7), ('14 hari', 14), ('30 hari', 30)]:
            if len(prices) >= window:
                past = prices[-window]
                if past > 0:
                    changes[label] = (prices[-1] - past) / past * 100

        if not changes:
            return None

        # Determine trend consistency
        trends = list(changes.values())
        all_up = all(t > 1 for t in trends)
        all_down = all(t < -1 for t in trends)
        accelerating = (len(trends) >= 2 and
                        abs(trends[0]) > abs(trends[-1]))

        trend_parts = []
        for label, change in changes.items():
            arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
            trend_parts.append(f"{label} terakhir {arrow} {change:+.1f}%")

        description = "Tren harga terkini: " + "; ".join(trend_parts) + ". "

        if all_up and pct_change > 0:
            description += (
                "Tren kenaikan konsisten dalam beberapa periode menunjukkan "
                "tekanan harga naik yang kuat di pasar."
            )
            impact = 'high' if accelerating else 'medium'
        elif all_down and pct_change < 0:
            description += (
                "Tren penurunan konsisten mengindikasikan pelemahan harga "
                "yang berkelanjutan, kemungkinan karena pasokan yang cukup."
            )
            impact = 'high' if accelerating else 'medium'
        elif accelerating:
            description += (
                "Pergerakan harga jangka pendek lebih cepat dari jangka panjang, "
                "menunjukkan adanya akselerasi perubahan harga."
            )
            impact = 'medium'
        else:
            description += (
                "Pergerakan harga relatif moderat tanpa tren dominan yang jelas."
            )
            impact = 'low'

        return {
            'name': '📈 Tren Harga Terkini',
            'impact': impact,
            'description': description,
        }

    def _analyze_seasonal(self, series, commodity, forecast_date, pct_change):
        """Analyze seasonal/harvest patterns."""
        target_month = forecast_date.month
        kb = self._get_commodity_kb(commodity)

        if not kb:
            return None

        is_harvest = target_month in kb.get('harvest_months', [])
        is_scarce = target_month in kb.get('scarce_months', [])

        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
        month_name = month_names.get(target_month, str(target_month))

        if is_harvest:
            description = (
                f"Bulan {month_name} merupakan periode MUSIM PANEN untuk "
                f"{commodity}. {kb['harvest_description']}"
            )
            # Harvest usually brings prices down
            if pct_change < 0:
                description += (
                    " Prediksi harga turun SESUAI dengan pola musim panen."
                )
                impact = 'high'
            else:
                description += (
                    " Namun prediksi menunjukkan harga naik BERTENTANGAN dengan "
                    "pola musim panen, kemungkinan ada faktor lain yang lebih kuat "
                    "(permintaan tinggi, gangguan distribusi, atau cuaca buruk)."
                )
                impact = 'high'

        elif is_scarce:
            description = (
                f"Bulan {month_name} merupakan periode MUSIM PACEKLIK untuk "
                f"{commodity}. {kb['scarce_description']}"
            )
            if pct_change > 0:
                description += (
                    " Prediksi kenaikan harga SESUAI dengan pola masa paceklik."
                )
                impact = 'high'
            else:
                description += (
                    " Prediksi harga turun BERTENTANGAN dengan pola paceklik, "
                    "kemungkinan ada intervensi pasar atau stok cadangan cukup."
                )
                impact = 'medium'
        else:
            # Transition period
            # Check proximity to harvest or scarce
            months_to_harvest = self._months_distance(
                target_month, kb.get('harvest_months', [])
            )
            months_to_scarce = self._months_distance(
                target_month, kb.get('scarce_months', [])
            )

            if months_to_harvest is not None and months_to_harvest <= 2:
                description = (
                    f"Bulan {month_name} merupakan periode transisi menjelang "
                    f"musim panen {commodity} yang akan datang dalam "
                    f"~{months_to_harvest} bulan. Pasar mulai mengantisipasi "
                    f"peningkatan pasokan."
                )
                impact = 'medium'
            elif months_to_scarce is not None and months_to_scarce <= 2:
                description = (
                    f"Bulan {month_name} merupakan periode transisi menjelang "
                    f"masa paceklik {commodity} dalam ~{months_to_scarce} bulan. "
                    f"Pedagang mungkin mulai menahan stok."
                )
                impact = 'medium'
            else:
                description = (
                    f"Bulan {month_name} merupakan periode normal untuk "
                    f"{commodity}, bukan musim panen maupun paceklik. "
                    f"Pasokan dan permintaan relatif seimbang."
                )
                impact = 'low'

            # Add historical monthly comparison
            series_with_month = series.copy()
            series_with_month['month'] = series_with_month['date'].dt.month
            month_data = series_with_month[
                series_with_month['month'] == target_month
            ]['price']
            if len(month_data) > 5:
                avg = month_data.mean()
                std = month_data.std()
                description += (
                    f" Rata-rata historis harga di bulan {month_name}: "
                    f"Rp {avg:,.0f}/kg (±Rp {std:,.0f})."
                )

        return {
            'name': '🌾 Faktor Musiman & Panen',
            'impact': impact,
            'description': description,
        }

    def _analyze_weather_season(self, commodity, forecast_date, pct_change):
        """Analyze weather season impact."""
        target_month = forecast_date.month
        kb = self._get_commodity_kb(commodity)

        if not kb:
            return None

        sensitivity = kb.get('weather_sensitivity', 'low')
        if sensitivity == 'low':
            # Still include but with low impact
            pass

        # Determine current weather season
        if target_month in self.WEATHER_SEASONS['rainy']['months']:
            season = self.WEATHER_SEASONS['rainy']
            season_key = 'rainy'
        else:
            season = self.WEATHER_SEASONS['dry']
            season_key = 'dry'

        description = (
            f"Periode prediksi jatuh pada {season['name']} "
            f"({season['description']}) "
        )

        # Add commodity-specific weather impact
        description += f"\n\n**Dampak terhadap {commodity}:** {kb['weather_description']}"

        sensitivity_labels = {
            'very_high': ('tinggi', 'high'),
            'high': ('tinggi', 'high'),
            'medium': ('sedang', 'medium'),
            'low': ('rendah', 'low'),
        }
        sens_label, impact = sensitivity_labels.get(
            sensitivity, ('rendah', 'low')
        )

        # Enhance impact based on weather-price alignment
        if sensitivity in ('very_high', 'high'):
            if season_key == 'rainy' and pct_change > 5:
                description += (
                    f" Musim hujan saat ini berpotensi mengurangi pasokan "
                    f"{commodity} karena risiko kerusakan tanaman."
                )
                impact = 'high'
            elif season_key == 'dry' and pct_change > 5:
                description += (
                    f" Kemarau dapat mengurangi produktivitas {commodity} "
                    f"dan mendorong kenaikan harga."
                )
                impact = 'high'

        description += (
            f"\n\nTingkat sensitivitas {commodity} terhadap cuaca: **{sens_label}**."
        )

        return {
            'name': '🌦️ Pengaruh Cuaca & Musim',
            'impact': impact,
            'description': description,
        }

    def _analyze_holiday_effect(self, commodity, forecast_date, pct_change):
        """Analyze proximity to major holidays and their price impact."""
        kb = self._get_commodity_kb(commodity)
        if not kb:
            return None

        sensitivity = kb.get('holiday_sensitivity', 'low')

        # Approximate Islamic holiday dates (shift ~11 days/year)
        # Using 2026 estimates
        year = forecast_date.year
        holidays = self._get_approximate_holidays(year)

        nearest_holiday = None
        nearest_days = float('inf')

        for hol_key, hol_dates in holidays.items():
            for hol_date in hol_dates:
                days_diff = (hol_date - forecast_date).days
                # Check if forecast is within impact window
                hol_config = self.HOLIDAY_CALENDAR.get(hol_key, {})
                pre = hol_config.get('pre_impact_days', 30)
                post = hol_config.get('post_impact_days', 7)

                if -post <= days_diff <= pre:
                    if abs(days_diff) < nearest_days:
                        nearest_days = abs(days_diff)
                        nearest_holiday = {
                            'key': hol_key,
                            'date': hol_date,
                            'days_diff': days_diff,
                            'config': hol_config,
                        }

        if nearest_holiday is None:
            # No nearby holiday
            description = (
                f"Tidak ada hari besar dalam periode yang berdekatan dengan "
                f"tanggal prediksi. Faktor hari raya tidak signifikan terhadap "
                f"harga {commodity} saat ini."
            )
            return {
                'name': '🕌 Efek Hari Raya',
                'impact': 'low',
                'description': description,
            }

        hol_name = nearest_holiday['config'].get('name', 'Hari Raya')
        days_diff = nearest_holiday['days_diff']
        hol_desc = nearest_holiday['config'].get('description', '')

        if days_diff > 0:
            timing = f"{days_diff} hari MENJELANG {hol_name}"
        elif days_diff < 0:
            timing = f"{abs(days_diff)} hari SETELAH {hol_name}"
        else:
            timing = f"bertepatan dengan {hol_name}"

        description = (
            f"Tanggal prediksi berjarak {timing}. "
            f"{hol_desc} "
        )

        # Add commodity-specific holiday description
        description += f"\n\n**Dampak terhadap {commodity}:** {kb['holiday_description']}"

        sensitivity_map = {
            'very_high': 'high',
            'high': 'high',
            'medium': 'medium',
            'low': 'low',
        }
        impact = sensitivity_map.get(sensitivity, 'low')

        # Increase impact if price movement aligns with holiday expectation
        if days_diff > 0 and pct_change > 5 and sensitivity in ('very_high', 'high'):
            description += (
                f" Kenaikan harga prediksi sesuai dengan pola historis "
                f"menjelang {hol_name}."
            )
            impact = 'high'

        return {
            'name': '🕌 Efek Hari Raya',
            'impact': impact,
            'description': description,
        }

    def _analyze_historical_anomaly(self, series, predicted_price, forecast_date):
        """Analyze if the prediction deviates from historical patterns."""
        target_month = forecast_date.month
        series_monthly = series.copy()
        series_monthly['month'] = series_monthly['date'].dt.month

        month_data = series_monthly[
            series_monthly['month'] == target_month
        ]['price']

        if len(month_data) < 5:
            return None

        hist_mean = month_data.mean()
        hist_std = month_data.std()
        hist_min = month_data.min()
        hist_max = month_data.max()

        if hist_std <= 0:
            hist_std = hist_mean * 0.05

        z_score = abs(predicted_price - hist_mean) / hist_std

        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
        month_name = month_names.get(target_month, str(target_month))

        description = (
            f"Berdasarkan data historis bulan {month_name}: "
            f"rata-rata Rp {hist_mean:,.0f}/kg, "
            f"rentang Rp {hist_min:,.0f} — Rp {hist_max:,.0f}/kg, "
            f"standar deviasi Rp {hist_std:,.0f}. "
            f"Harga prediksi Rp {predicted_price:,.0f}/kg "
        )

        if z_score > 3:
            description += (
                f"menyimpang SANGAT JAUH dari pola historis (z-score: {z_score:.1f}). "
                "Ini merupakan ANOMALI BESAR yang mengindikasikan adanya "
                "kejadian luar biasa di pasar — seperti gangguan pasokan besar, "
                "perubahan kebijakan, atau krisis."
            )
            impact = 'high'
        elif z_score > 2:
            description += (
                f"menyimpang cukup signifikan dari pola historis (z-score: {z_score:.1f}). "
                "Ini menunjukkan kondisi pasar yang tidak biasa untuk periode ini."
            )
            impact = 'high'
        elif z_score > 1:
            description += (
                f"sedikit menyimpang dari pola historis (z-score: {z_score:.1f}). "
                "Deviasi ini masih dalam batas wajar namun perlu diperhatikan."
            )
            impact = 'medium'
        else:
            description += (
                f"SESUAI dengan pola historis (z-score: {z_score:.1f}). "
                "Harga prediksi berada dalam rentang normal untuk periode ini."
            )
            impact = 'low'

        return {
            'name': '📊 Anomali Historis',
            'impact': impact,
            'description': description,
        }

    def _analyze_regional_factor(self, province, commodity, current_price,
                                 predicted_price):
        """Analyze regional price comparison."""
        national = self.df[self.df['commodity'] == commodity]
        if national.empty:
            return None

        latest_date = national['date'].max()
        latest_prices = national[national['date'] == latest_date]

        if latest_prices.empty or len(latest_prices) < 3:
            return None

        nat_avg = latest_prices['price'].mean()
        nat_std = latest_prices['price'].std()
        nat_min = latest_prices['price'].min()
        nat_max = latest_prices['price'].max()
        cheapest_prov = latest_prices.loc[
            latest_prices['price'].idxmin(), 'province'
        ]
        most_expensive_prov = latest_prices.loc[
            latest_prices['price'].idxmax(), 'province'
        ]

        if nat_std <= 0:
            nat_std = nat_avg * 0.05

        price_vs_national = ((current_price - nat_avg) / nat_avg * 100
                             if nat_avg > 0 else 0)

        description = (
            f"**Perbandingan Regional {commodity}:**\n"
            f"- Rata-rata nasional: Rp {nat_avg:,.0f}/kg\n"
            f"- Harga di {province}: Rp {current_price:,.0f}/kg "
            f"({price_vs_national:+.1f}% vs nasional)\n"
            f"- Termurah: {cheapest_prov} (Rp {nat_min:,.0f}/kg)\n"
            f"- Termahal: {most_expensive_prov} (Rp {nat_max:,.0f}/kg)\n\n"
        )

        if abs(price_vs_national) > 20:
            description += (
                f"Harga di {province} menyimpang SIGNIFIKAN dari rata-rata "
                f"nasional. Ini bisa disebabkan oleh faktor logistik, "
                f"ketersediaan pasokan lokal, atau biaya distribusi yang tinggi."
            )
            impact = 'high'
        elif abs(price_vs_national) > 10:
            description += (
                f"Harga di {province} cukup berbeda dari rata-rata nasional, "
                f"menunjukkan adanya faktor regional yang mempengaruhi harga."
            )
            impact = 'medium'
        else:
            description += (
                f"Harga di {province} relatif sesuai dengan rata-rata nasional, "
                f"menunjukkan kondisi pasar yang merata."
            )
            impact = 'low'

        return {
            'name': '🗺️ Faktor Regional',
            'impact': impact,
            'description': description,
        }

    # ------------------------------------------------------------------ #
    # Narrative builders
    # ------------------------------------------------------------------ #

    def _build_summary(self, direction, pct_change, commodity, province,
                       forecast_date, factors):
        """Build a 1-2 sentence summary."""
        direction_text = {
            'NAIK': f'naik sebesar {abs(pct_change):.1f}%',
            'TURUN': f'turun sebesar {abs(pct_change):.1f}%',
            'STABIL': f'relatif stabil (perubahan {abs(pct_change):.1f}%)',
        }

        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }

        summary = (
            f"Harga {commodity} di {province} diprediksi "
            f"{direction_text.get(direction, 'berubah')} "
            f"pada {forecast_date.day} {month_names.get(forecast_date.month, '')} "
            f"{forecast_date.year}."
        )

        # Add top factor
        high_factors = [f for f in factors if f['impact'] == 'high']
        if high_factors:
            top_names = [f['name'].split(' ', 1)[-1] for f in high_factors[:2]]
            summary += (
                f" Faktor utama: {' dan '.join(top_names)}."
            )

        return summary

    def _build_narrative(self, direction, pct_change, commodity, province,
                         forecast_date, current_price, predicted_price, factors):
        """Build a full narrative paragraph."""
        month_names = {
            1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
            5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
            9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
        }
        month_name = month_names.get(forecast_date.month, '')

        direction_verb = {
            'NAIK': 'mengalami kenaikan',
            'TURUN': 'mengalami penurunan',
            'STABIL': 'relatif stabil',
        }

        narrative = (
            f"Berdasarkan analisis AI multi-model, harga {commodity} di "
            f"{province} diprediksi {direction_verb.get(direction, 'berubah')} "
            f"dari Rp {current_price:,.0f}/kg menjadi Rp {predicted_price:,.0f}/kg "
            f"(perubahan {pct_change:+.1f}%) pada "
            f"{forecast_date.day} {month_name} {forecast_date.year}.\n\n"
        )

        # Group factors by impact
        high_factors = [f for f in factors if f['impact'] == 'high']
        medium_factors = [f for f in factors if f['impact'] == 'medium']
        low_factors = [f for f in factors if f['impact'] == 'low']

        if high_factors:
            narrative += "**Faktor-faktor UTAMA yang mendorong prediksi ini:**\n\n"
            for f in high_factors:
                narrative += f"🔴 **{f['name']}**: {f['description']}\n\n"

        if medium_factors:
            narrative += "**Faktor pendukung:**\n\n"
            for f in medium_factors:
                narrative += f"🟡 **{f['name']}**: {f['description']}\n\n"

        if low_factors:
            narrative += "**Faktor minor:**\n\n"
            for f in low_factors:
                narrative += f"🟢 **{f['name']}**: {f['description']}\n\n"

        # Add conclusion
        if direction == 'NAIK':
            narrative += (
                "**Kesimpulan:** Kombinasi faktor-faktor di atas mendorong "
                "tekanan kenaikan harga. Disarankan untuk mempersiapkan "
                "langkah antisipasi seperti operasi pasar atau pelepasan "
                "stok cadangan jika kenaikan berlanjut."
            )
        elif direction == 'TURUN':
            narrative += (
                "**Kesimpulan:** Faktor-faktor di atas mengindikasikan "
                "pelemahan harga. Ini bisa menjadi kabar baik bagi konsumen "
                "namun perlu diantisipasi dampaknya terhadap kesejahteraan "
                "petani/produsen."
            )
        else:
            narrative += (
                "**Kesimpulan:** Kondisi pasar relatif seimbang. Tidak "
                "diperlukan intervensi khusus, namun tetap perlu monitoring "
                "berkelanjutan terhadap faktor-faktor yang berpotensi "
                "mengubah keseimbangan."
            )

        return narrative

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_commodity_kb(self, commodity):
        """Find best matching commodity in knowledge base."""
        commodity_lower = commodity.lower()

        # Direct match
        for key in self.COMMODITY_KB:
            if key.lower() in commodity_lower or commodity_lower in key.lower():
                return self.COMMODITY_KB[key]

        # Keyword matching
        keyword_map = {
            'beras': 'Beras',
            'cabai merah': 'Cabai Merah',
            'cabai rawit': 'Cabai Rawit',
            'bawang merah': 'Bawang Merah',
            'bawang putih': 'Bawang Putih',
            'daging ayam': 'Daging Ayam',
            'ayam': 'Daging Ayam',
            'daging sapi': 'Daging Sapi',
            'sapi': 'Daging Sapi',
            'telur': 'Telur Ayam',
            'minyak goreng': 'Minyak Goreng',
            'minyak': 'Minyak Goreng',
            'gula': 'Gula Pasir',
        }
        for keyword, kb_key in keyword_map.items():
            if keyword in commodity_lower:
                return self.COMMODITY_KB[kb_key]

        return None

    def _months_distance(self, current_month, target_months):
        """Calculate minimum month distance to nearest target month."""
        if not target_months:
            return None
        distances = []
        for t in target_months:
            d = (t - current_month) % 12
            distances.append(d)
        return min(distances) if distances else None

    def _get_approximate_holidays(self, year):
        """Get approximate major holiday dates for a given year.

        Note: Islamic holidays shift ~11 days earlier each Gregorian year.
        These are rough approximations for planning purposes.
        """
        # Base: 2026 Ramadan starts ~Feb 18, Eid al-Fitr ~Mar 20
        # Idul Adha ~May 27, 2026
        # Adjustments for different years
        year_offset = year - 2026
        day_shift = int(year_offset * -10.63)  # ~10.63 days/year shift

        def _shift_date(base_month, base_day):
            try:
                base = pd.Timestamp(year=year, month=base_month, day=base_day)
                shifted = base + pd.Timedelta(days=day_shift)
                return shifted
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                return pd.Timestamp(year=year, month=base_month, day=base_day)

        return {
            'ramadan': [
                _shift_date(3, 20),  # Eid al-Fitr (end of Ramadan)
            ],
            'idul_adha': [
                _shift_date(5, 27),  # Eid al-Adha
            ],
            'natal_tahun_baru': [
                pd.Timestamp(year=year, month=12, day=25),  # Christmas
                pd.Timestamp(year=year, month=1, day=1),    # New Year
            ],
        }

    def _fallback_narrative(self, predicted_price):
        """Return a fallback when data is insufficient."""
        return {
            'direction': 'UNKNOWN',
            'pct_change': 0,
            'summary': (
                'Data historis tidak cukup untuk menghasilkan analisis '
                'naratif yang akurat. Diperlukan minimal 14 hari data.'
            ),
            'factors': [],
            'narrative': (
                'Analisis penyebab tidak tersedia karena data historis '
                'tidak mencukupi. Kumpulkan lebih banyak data untuk '
                'mendapatkan analisis yang lebih komprehensif.'
            ),
        }
