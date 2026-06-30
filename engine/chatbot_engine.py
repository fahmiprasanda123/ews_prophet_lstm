"""
Chatbot Engine for Agri-AI EWS.
Rule-based + data-driven chatbot that answers questions about Indonesian
food commodity prices using the existing DataStore.
"""
import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from data.database import get_store

logger = logging.getLogger(__name__)


class ChatbotEngine:
    """Rule-based chatbot engine with DataStore integration.

    Recognises user intents via regex patterns and queries the SQLite
    database to produce data-driven answers in Bahasa Indonesia.
    """

    # ------------------------------------------------------------------ #
    # Commodity & province aliases (case-insensitive matching helpers)
    # ------------------------------------------------------------------ #
    COMMODITY_ALIASES = {
        "beras": "Beras",
        "beras medium": "Beras",
        "beras premium": "Beras",
        "cabai": "Cabai Merah",
        "cabai merah": "Cabai Merah",
        "cabai merah besar": "Cabai Merah",
        "cabe": "Cabai Merah",
        "cabe merah": "Cabai Merah",
        "cabai rawit": "Cabai Rawit",
        "cabai rawit merah": "Cabai Rawit",
        "cabe rawit": "Cabai Rawit",
        "rawit": "Cabai Rawit",
        "bawang merah": "Bawang Merah",
        "bawang putih": "Bawang Putih",
        "gula": "Gula Pasir",
        "gula pasir": "Gula Pasir",
        "minyak": "Minyak Goreng",
        "minyak goreng": "Minyak Goreng",
        "migor": "Minyak Goreng",
        "daging": "Daging Ayam",
        "ayam": "Daging Ayam",
        "daging ayam": "Daging Ayam",
        "daging ayam ras": "Daging Ayam",
        "daging sapi": "Daging Sapi",
        "sapi": "Daging Sapi",
        "telur": "Telur Ayam",
        "telur ayam": "Telur Ayam",
        "telur ayam ras": "Telur Ayam",
        "telor": "Telur Ayam",
    }

    PROVINCE_ALIASES = {
        "jakarta": "DKI Jakarta",
        "dki jakarta": "DKI Jakarta",
        "dki": "DKI Jakarta",
        "jawa barat": "Jawa Barat",
        "jabar": "Jawa Barat",
        "jawa tengah": "Jawa Tengah",
        "jateng": "Jawa Tengah",
        "jawa timur": "Jawa Timur",
        "jatim": "Jawa Timur",
        "surabaya": "Jawa Timur",
        "bandung": "Jawa Barat",
        "semarang": "Jawa Tengah",
        "yogyakarta": "DI Yogyakarta",
        "jogja": "DI Yogyakarta",
        "di yogyakarta": "DI Yogyakarta",
        "bali": "Bali",
        "sumatera utara": "Sumatera Utara",
        "sumut": "Sumatera Utara",
        "medan": "Sumatera Utara",
        "sumatera barat": "Sumatera Barat",
        "sumbar": "Sumatera Barat",
        "sumatera selatan": "Sumatera Selatan",
        "sumsel": "Sumatera Selatan",
        "palembang": "Sumatera Selatan",
        "kalimantan barat": "Kalimantan Barat",
        "kalbar": "Kalimantan Barat",
        "kalimantan timur": "Kalimantan Timur",
        "kaltim": "Kalimantan Timur",
        "kalimantan selatan": "Kalimantan Selatan",
        "kalsel": "Kalimantan Selatan",
        "kalimantan tengah": "Kalimantan Tengah",
        "kalteng": "Kalimantan Tengah",
        "kalimantan utara": "Kalimantan Utara",
        "kaltara": "Kalimantan Utara",
        "sulawesi selatan": "Sulawesi Selatan",
        "sulsel": "Sulawesi Selatan",
        "makassar": "Sulawesi Selatan",
        "sulawesi utara": "Sulawesi Utara",
        "sulut": "Sulawesi Utara",
        "sulawesi tengah": "Sulawesi Tengah",
        "sulteng": "Sulawesi Tengah",
        "sulawesi tenggara": "Sulawesi Tenggara",
        "sultra": "Sulawesi Tenggara",
        "sulawesi barat": "Sulawesi Barat",
        "sulbar": "Sulawesi Barat",
        "papua": "Papua",
        "papua barat": "Papua Barat",
        "maluku": "Maluku",
        "maluku utara": "Maluku Utara",
        "ntt": "Nusa Tenggara Timur",
        "nusa tenggara timur": "Nusa Tenggara Timur",
        "ntb": "Nusa Tenggara Barat",
        "nusa tenggara barat": "Nusa Tenggara Barat",
        "riau": "Riau",
        "kepulauan riau": "Kepulauan Riau",
        "kepri": "Kepulauan Riau",
        "jambi": "Jambi",
        "bengkulu": "Bengkulu",
        "lampung": "Lampung",
        "bangka belitung": "Kepulauan Bangka Belitung",
        "babel": "Kepulauan Bangka Belitung",
        "aceh": "Aceh",
        "gorontalo": "Gorontalo",
        "banten": "Banten",
    }

    # ------------------------------------------------------------------ #
    # Intent patterns
    # ------------------------------------------------------------------ #
    INTENT_PATTERNS = [
        # Greetings
        ("salam", re.compile(
            r"\b(halo|hai|hi|hello|hey|selamat\s+(pagi|siang|sore|malam)|assalamualaikum|apa\s*kabar)\b",
            re.IGNORECASE,
        )),
        # Help
        ("bantuan", re.compile(
            r"\b(help|bantuan|bantu|bisa\s+apa|fitur|kemampuan|apa\s+yang\s+bisa|cara\s+pakai|panduan)\b",
            re.IGNORECASE,
        )),
        # Compare
        ("bandingkan", re.compile(
            r"\b(bandingkan|banding|perbandingan|komparasi|compare|vs|versus|dibanding)\b",
            re.IGNORECASE,
        )),
        # Statistics
        ("statistik", re.compile(
            r"\b(statistik|rata[\s-]?rata|average|mean|min|max|tertinggi|terendah|std|deviasi|median|rangkuman|summary|rekap)\b",
            re.IGNORECASE,
        )),
        # Prediction / Forecast — MUST be before harga_terbaru
        # Uses \w* after stems to match Indonesian suffixes: -nya, -kan, -mu, etc.
        ("prediksi", re.compile(
            r"\b(prediksi\w*|forecast\w*|ramalan\w*|ramal\w*|predik\w*|perkiraan\w*|estimasi\w*|proyeksi\w*|harga\s+besok|minggu\s+depan|bulan\s+depan|akan\s+datang|masa\s+depan|kedepan\w*|ke\s+depan)",
            re.IGNORECASE,
        )),
        # Trend
        ("tren", re.compile(
            r"\b(tren|trend|naik|turun|kenaikan|penurunan|perubahan|fluktuasi|volatil|grafik|chart|perkembangan|pergerakan)\b",
            re.IGNORECASE,
        )),
        # Latest price
        ("harga_terbaru", re.compile(
            r"\b(harga|berapa\s+harga|price|biaya|tarif|terbaru|sekarang|saat\s+ini|hari\s+ini|terakhir|terkini|latest)\b",
            re.IGNORECASE,
        )),
        # Commodity list
        ("daftar_komoditas", re.compile(
            r"\b(daftar|list|komoditas\s+apa|jenis|macam|apa\s+saja\s+komoditas|komoditas\s+tersedia)\b",
            re.IGNORECASE,
        )),
        # Province list
        ("daftar_provinsi", re.compile(
            r"\b(provinsi|wilayah|daerah|region|area|kota)\b",
            re.IGNORECASE,
        )),
    ]

    def __init__(self):
        self.store = get_store()
        # Cache province and commodity lists
        self._provinces = None
        self._commodities = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, user_message: str) -> dict:
        """Process a user message and return a response.

        Returns:
            dict with keys:
                - text: str  (main response text, may include markdown)
                - chart_data: pd.DataFrame or None
                - chart_type: str or None  ('line', 'bar', etc.)
                - chart_title: str or None
        """
        msg = user_message.strip()
        if not msg:
            return self._response("Silakan ketik pertanyaan Anda 😊")

        # Detect intent
        intent = self._detect_intent(msg)

        # Extract entities
        commodity = self._extract_commodity(msg)
        provinces = self._extract_provinces(msg)

        # Route to handler
        handlers = {
            "salam": self._handle_salam,
            "bantuan": self._handle_bantuan,
            "daftar_komoditas": self._handle_daftar_komoditas,
            "daftar_provinsi": self._handle_daftar_provinsi,
            "harga_terbaru": self._handle_harga_terbaru,
            "prediksi": self._handle_prediksi,
            "tren": self._handle_tren,
            "statistik": self._handle_statistik,
            "bandingkan": self._handle_bandingkan,
        }

        handler = handlers.get(intent, self._handle_unknown)
        return handler(msg, commodity, provinces)

    # ------------------------------------------------------------------ #
    # Intent detection
    # ------------------------------------------------------------------ #
    def _detect_intent(self, msg: str) -> str:
        """Detect the user's intent from their message."""
        for intent_name, pattern in self.INTENT_PATTERNS:
            if pattern.search(msg):
                return intent_name
        return "unknown"

    # ------------------------------------------------------------------ #
    # Entity extraction
    # ------------------------------------------------------------------ #
    def _extract_commodity(self, msg: str) -> Optional[str]:
        """Try to find a commodity name in the message."""
        msg_lower = msg.lower()
        # Try longest alias first to avoid partial matches
        for alias in sorted(self.COMMODITY_ALIASES.keys(), key=len, reverse=True):
            if alias in msg_lower:
                return self.COMMODITY_ALIASES[alias]

        # Fuzzy fallback: check against actual DB commodity names
        for comm in self._get_commodities():
            if comm.lower() in msg_lower:
                return comm
        return None

    def _extract_provinces(self, msg: str) -> List[str]:
        """Try to find province names in the message. May return multiple."""
        msg_lower = msg.lower()
        found = []
        seen = set()
        for alias in sorted(self.PROVINCE_ALIASES.keys(), key=len, reverse=True):
            if alias in msg_lower:
                prov = self.PROVINCE_ALIASES[alias]
                if prov not in seen:
                    found.append(prov)
                    seen.add(prov)

        # Also check actual DB province names
        for prov in self._get_provinces():
            if prov.lower() in msg_lower and prov not in seen:
                found.append(prov)
                seen.add(prov)
        return found

    # ------------------------------------------------------------------ #
    # Cached data helpers
    # ------------------------------------------------------------------ #
    def _get_provinces(self) -> list[str]:
        if self._provinces is None:
            self._provinces = self.store.get_provinces()
        return self._provinces

    def _get_commodities(self) -> list[str]:
        if self._commodities is None:
            self._commodities = self.store.get_commodities()
        return self._commodities

    # ------------------------------------------------------------------ #
    # Response builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _response(text: str, chart_data=None, chart_type=None, chart_title=None) -> dict:
        return {
            "text": text,
            "chart_data": chart_data,
            "chart_type": chart_type,
            "chart_title": chart_title,
        }

    @staticmethod
    def _fmt_price(price: float) -> str:
        """Format price as Indonesian Rupiah."""
        return f"Rp {price:,.0f}".replace(",", ".")

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #
    def _handle_salam(self, msg, commodity, provinces):
        hour = datetime.now().hour
        if hour < 11:
            greeting = "Selamat pagi! ☀️"
        elif hour < 15:
            greeting = "Selamat siang! 🌤️"
        elif hour < 18:
            greeting = "Selamat sore! 🌅"
        else:
            greeting = "Selamat malam! 🌙"

        return self._response(
            f"{greeting}\n\n"
            f"Saya adalah **Agri-AI Chatbot** 🤖, asisten virtual untuk sistem peringatan dini harga pangan.\n\n"
            f"Saya bisa membantu Anda dengan:\n"
            f"- 💰 Cek harga komoditas terbaru\n"
            f"- 🔮 Prediksi harga dengan AI (Prophet + LSTM)\n"
            f"- 📈 Analisis tren harga\n"
            f"- 📊 Statistik & perbandingan harga\n"
            f"- 🗺️ Informasi provinsi & komoditas\n\n"
            f"Silakan tanyakan apa saja! Contoh: *\"Prediksi harga beras di Jakarta\"*"
        )

    def _handle_bantuan(self, msg, commodity, provinces):
        return self._response(
            "## 📖 Panduan Penggunaan Chatbot\n\n"
            "Berikut contoh pertanyaan yang bisa Anda ajukan:\n\n"
            "| Kategori | Contoh Pertanyaan |\n"
            "|----------|------------------|\n"
            "| 💰 **Harga** | *\"Berapa harga beras di Jakarta?\"* |\n"
            "| 🔮 **Prediksi** | *\"Prediksi harga cabai di Jakarta 30 hari\"* |\n"
            "| 📈 **Tren** | *\"Bagaimana tren harga cabai?\"* |\n"
            "| 📊 **Statistik** | *\"Rata-rata harga bawang merah\"* |\n"
            "| 🔄 **Perbandingan** | *\"Bandingkan harga beras Jakarta dan Jawa Barat\"* |\n"
            "| 📋 **Daftar** | *\"Komoditas apa saja yang tersedia?\"* |\n"
            "| 🗺️ **Wilayah** | *\"Provinsi mana saja?\"* |\n\n"
            "> 💡 **Tips**: Semakin spesifik pertanyaan Anda, semakin akurat jawaban saya!"
        )

    def _handle_daftar_komoditas(self, msg, commodity, provinces):
        commodities = self._get_commodities()
        if not commodities:
            return self._response("⚠️ Belum ada data komoditas di database.")

        items = "\n".join(f"- {c}" for c in commodities)
        return self._response(
            f"## 🌽 Daftar Komoditas Tersedia\n\n"
            f"Terdapat **{len(commodities)} komoditas** yang dipantau:\n\n"
            f"{items}\n\n"
            f"Tanyakan harga spesifik, misalnya: *\"Berapa harga cabai di Jawa Timur?\"*"
        )

    def _handle_daftar_provinsi(self, msg, commodity, provinces):
        provs = self._get_provinces()
        if not provs:
            return self._response("⚠️ Belum ada data provinsi di database.")

        # Show in 2-column format
        mid = (len(provs) + 1) // 2
        col1 = provs[:mid]
        col2 = provs[mid:]

        rows = []
        for i in range(max(len(col1), len(col2))):
            c1 = col1[i] if i < len(col1) else ""
            c2 = col2[i] if i < len(col2) else ""
            rows.append(f"| {c1} | {c2} |")

        table = "| Provinsi | Provinsi |\n|----------|----------|\n" + "\n".join(rows)

        return self._response(
            f"## 🗺️ Daftar Provinsi\n\n"
            f"Data tersedia untuk **{len(provs)} provinsi**:\n\n"
            f"{table}"
        )

    def _handle_harga_terbaru(self, msg, commodity, provinces):
        if not commodity and not provinces:
            # General latest prices overview
            latest = self.store.get_latest_prices()
            if latest.empty:
                return self._response("⚠️ Belum ada data harga di database.")

            # Show summary per commodity (national average)
            summary = latest.groupby("commodity")["price"].agg(["mean", "min", "max"]).reset_index()
            summary.columns = ["Komoditas", "Rata-rata", "Terendah", "Tertinggi"]

            rows = []
            for _, r in summary.iterrows():
                rows.append(
                    f"| {r['Komoditas']} | {self._fmt_price(r['Rata-rata'])} | "
                    f"{self._fmt_price(r['Terendah'])} | {self._fmt_price(r['Tertinggi'])} |"
                )

            table = (
                "| Komoditas | Rata-rata | Terendah | Tertinggi |\n"
                "|-----------|-----------|----------|----------|\n"
                + "\n".join(rows)
            )

            return self._response(
                f"## 💰 Ringkasan Harga Terbaru (Nasional)\n\n{table}\n\n"
                f"Untuk detail spesifik, coba: *\"Harga beras di Jakarta\"*"
            )

        if commodity and provinces:
            province = provinces[0]
            series = self.store.get_series(province, commodity)
            if series.empty:
                return self._response(
                    f"⚠️ Tidak ditemukan data harga **{commodity}** di **{province}**."
                )

            latest_price = series["price"].iloc[-1]
            latest_date = series["date"].iloc[-1].strftime("%d %B %Y")

            # Calculate change
            change_text = ""
            if len(series) >= 2:
                prev_price = series["price"].iloc[-2]
                change = latest_price - prev_price
                pct = (change / prev_price) * 100
                if change > 0:
                    change_text = f"📈 Naik **{self._fmt_price(abs(change))}** (+{pct:.1f}%) dari data sebelumnya"
                elif change < 0:
                    change_text = f"📉 Turun **{self._fmt_price(abs(change))}** ({pct:.1f}%) dari data sebelumnya"
                else:
                    change_text = "➡️ Stabil, tidak ada perubahan dari data sebelumnya"

            return self._response(
                f"## 💰 Harga {commodity} di {province}\n\n"
                f"**{self._fmt_price(latest_price)}** per kg\n\n"
                f"📅 Data terakhir: {latest_date}\n\n"
                f"{change_text}",
                chart_data=series.tail(30),
                chart_type="line",
                chart_title=f"Harga {commodity} di {province} (30 hari terakhir)",
            )

        if commodity:
            latest = self.store.get_latest_prices(commodity)
            if latest.empty:
                return self._response(f"⚠️ Tidak ditemukan data harga **{commodity}**.")

            avg_price = latest["price"].mean()
            min_row = latest.loc[latest["price"].idxmin()]
            max_row = latest.loc[latest["price"].idxmax()]

            return self._response(
                f"## 💰 Harga {commodity} (Nasional)\n\n"
                f"- 📊 Rata-rata nasional: **{self._fmt_price(avg_price)}**\n"
                f"- 📉 Terendah: **{self._fmt_price(min_row['price'])}** ({min_row['province']})\n"
                f"- 📈 Tertinggi: **{self._fmt_price(max_row['price'])}** ({max_row['province']})\n\n"
                f"Untuk melihat di provinsi tertentu, coba: *\"Harga {commodity.lower()} di Jakarta\"*"
            )

        if provinces:
            province = provinces[0]
            latest = self.store.get_latest_prices()
            prov_data = latest[latest["province"] == province]
            if prov_data.empty:
                return self._response(
                    f"⚠️ Tidak ditemukan data harga untuk **{province}**."
                )

            rows = []
            for _, r in prov_data.iterrows():
                rows.append(f"| {r['commodity']} | {self._fmt_price(r['price'])} |")

            table = "| Komoditas | Harga |\n|-----------|-------|\n" + "\n".join(rows)

            return self._response(
                f"## 💰 Harga Komoditas di {province}\n\n{table}"
            )

        return self._response("Silakan sebutkan komoditas atau provinsi yang ingin dicek harganya.")

    def _handle_tren(self, msg, commodity, provinces):
        if not commodity:
            return self._response(
                "📈 Untuk melihat tren harga, sebutkan **komoditas** yang ingin dianalisis.\n\n"
                "Contoh: *\"Tren harga cabai\"* atau *\"Bagaimana perkembangan harga beras di Jakarta?\"*"
            )

        province = provinces[0] if provinces else None

        if province:
            series = self.store.get_series(province, commodity)
        else:
            # National average
            all_data = self.store.load_all()
            if commodity:
                all_data = all_data[all_data["commodity"] == commodity]
            series = all_data.groupby("date")["price"].mean().reset_index()

        if series.empty:
            loc_text = f" di {province}" if province else " (nasional)"
            return self._response(f"⚠️ Tidak ditemukan data tren **{commodity}**{loc_text}.")

        # Analyze last 30 and 90 days
        recent_30 = series.tail(30)
        recent_90 = series.tail(90)

        if len(recent_30) >= 2:
            start_price = recent_30["price"].iloc[0]
            end_price = recent_30["price"].iloc[-1]
            change_30 = ((end_price - start_price) / start_price) * 100
        else:
            change_30 = 0

        if len(recent_90) >= 2:
            start_price_90 = recent_90["price"].iloc[0]
            end_price_90 = recent_90["price"].iloc[-1]
            change_90 = ((end_price_90 - start_price_90) / start_price_90) * 100
        else:
            change_90 = 0

        # Trend description
        if change_30 > 5:
            trend_emoji = "🔴"
            trend_desc = "**naik signifikan**"
        elif change_30 > 0:
            trend_emoji = "🟡"
            trend_desc = "**naik sedikit**"
        elif change_30 > -5:
            trend_emoji = "🟢"
            trend_desc = "**turun sedikit**"
        else:
            trend_emoji = "🟢"
            trend_desc = "**turun signifikan**"

        volatility = recent_30["price"].std() / recent_30["price"].mean() * 100
        vol_desc = "tinggi ⚠️" if volatility > 10 else "sedang" if volatility > 5 else "rendah ✅"

        loc_text = f" di {province}" if province else " (Nasional)"

        return self._response(
            f"## 📈 Analisis Tren {commodity}{loc_text}\n\n"
            f"### Perubahan Harga\n"
            f"- {trend_emoji} **30 hari terakhir**: Harga {trend_desc} ({change_30:+.1f}%)\n"
            f"- {'📈' if change_90 > 0 else '📉'} **90 hari terakhir**: {change_90:+.1f}%\n\n"
            f"### Volatilitas\n"
            f"- Tingkat volatilitas: **{vol_desc}** (CoV: {volatility:.1f}%)\n\n"
            f"### Harga Saat Ini\n"
            f"- Terakhir: **{self._fmt_price(end_price)}**\n"
            f"- Rata-rata 30 hari: **{self._fmt_price(recent_30['price'].mean())}**",
            chart_data=recent_90,
            chart_type="line",
            chart_title=f"Tren Harga {commodity}{loc_text} (90 hari)",
        )

    def _handle_statistik(self, msg, commodity, provinces):
        if not commodity:
            return self._response(
                "📊 Untuk melihat statistik, sebutkan **komoditas** yang ingin dianalisis.\n\n"
                "Contoh: *\"Statistik harga beras\"* atau *\"Rata-rata harga cabai di Jakarta\"*"
            )

        province = provinces[0] if provinces else None

        if province:
            series = self.store.get_series(province, commodity)
        else:
            all_data = self.store.load_all()
            if commodity:
                all_data = all_data[all_data["commodity"] == commodity]
            series = all_data.groupby("date")["price"].mean().reset_index()

        if series.empty:
            loc_text = f" di {province}" if province else ""
            return self._response(f"⚠️ Tidak ditemukan data **{commodity}**{loc_text}.")

        prices = series["price"]
        loc_text = f" di {province}" if province else " (Nasional)"

        # Monthly breakdown if enough data
        monthly_text = ""
        if len(series) > 60:
            series_copy = series.copy()
            series_copy["month"] = series_copy["date"].dt.to_period("M")
            monthly = series_copy.groupby("month")["price"].mean().tail(6)
            monthly_rows = []
            for period, avg in monthly.items():
                monthly_rows.append(f"| {period} | {self._fmt_price(avg)} |")
            monthly_text = (
                "\n### 📅 Rata-rata Bulanan (6 bulan terakhir)\n\n"
                "| Bulan | Rata-rata |\n|-------|----------|\n"
                + "\n".join(monthly_rows)
            )

        return self._response(
            f"## 📊 Statistik {commodity}{loc_text}\n\n"
            f"| Metrik | Nilai |\n"
            f"|--------|-------|\n"
            f"| Jumlah data | {len(prices):,} |\n"
            f"| Rata-rata | {self._fmt_price(prices.mean())} |\n"
            f"| Median | {self._fmt_price(prices.median())} |\n"
            f"| Minimum | {self._fmt_price(prices.min())} |\n"
            f"| Maximum | {self._fmt_price(prices.max())} |\n"
            f"| Std. Deviasi | {self._fmt_price(prices.std())} |\n"
            f"{monthly_text}",
            chart_data=series.tail(90),
            chart_type="line",
            chart_title=f"Harga {commodity}{loc_text} (90 hari terakhir)",
        )

    def _handle_bandingkan(self, msg, commodity, provinces):
        if not commodity:
            return self._response(
                "🔄 Untuk membandingkan harga, sebutkan **komoditas** dan **dua provinsi**.\n\n"
                "Contoh: *\"Bandingkan harga beras Jakarta dan Jawa Barat\"*"
            )

        if len(provinces) < 2:
            return self._response(
                f"🔄 Untuk membandingkan harga **{commodity}**, sebutkan **dua provinsi**.\n\n"
                f"Contoh: *\"Bandingkan harga {commodity.lower()} Jakarta dan Jawa Barat\"*"
            )

        prov1, prov2 = provinces[0], provinces[1]
        series1 = self.store.get_series(prov1, commodity)
        series2 = self.store.get_series(prov2, commodity)

        if series1.empty and series2.empty:
            return self._response(
                f"⚠️ Tidak ditemukan data **{commodity}** untuk {prov1} maupun {prov2}."
            )

        rows = []
        metrics = ["Harga terakhir", "Rata-rata", "Minimum", "Maximum"]

        def _safe_stats(s):
            if s.empty:
                return ["N/A"] * 4
            p = s["price"]
            return [
                self._fmt_price(p.iloc[-1]),
                self._fmt_price(p.mean()),
                self._fmt_price(p.min()),
                self._fmt_price(p.max()),
            ]

        stats1 = _safe_stats(series1)
        stats2 = _safe_stats(series2)

        for metric, v1, v2 in zip(metrics, stats1, stats2):
            rows.append(f"| {metric} | {v1} | {v2} |")

        table = (
            f"| Metrik | {prov1} | {prov2} |\n"
            f"|--------|{'---' * len(prov1)}|{'---' * len(prov2)}|\n"
            + "\n".join(rows)
        )

        # Difference summary
        diff_text = ""
        if not series1.empty and not series2.empty:
            p1 = series1["price"].iloc[-1]
            p2 = series2["price"].iloc[-1]
            diff = abs(p1 - p2)
            higher = prov1 if p1 > p2 else prov2
            pct = (diff / min(p1, p2)) * 100
            diff_text = (
                f"\n### 📌 Kesimpulan\n"
                f"Harga **{commodity}** di **{higher}** lebih tinggi "
                f"**{self._fmt_price(diff)}** ({pct:.1f}%) dibanding provinsi lainnya."
            )

        # Build combined chart data
        chart_data = None
        if not series1.empty and not series2.empty:
            s1 = series1.tail(90).copy()
            s2 = series2.tail(90).copy()
            s1["province"] = prov1
            s2["province"] = prov2
            chart_data = pd.concat([s1, s2], ignore_index=True)

        return self._response(
            f"## 🔄 Perbandingan Harga {commodity}\n\n{table}\n{diff_text}",
            chart_data=chart_data,
            chart_type="compare_line",
            chart_title=f"Perbandingan {commodity}: {prov1} vs {prov2}",
        )

    def _handle_prediksi(self, msg, commodity, provinces):
        """Handle price prediction/forecast requests using Prophet model."""
        if not commodity:
            return self._response(
                "🔮 Untuk prediksi harga, sebutkan **komoditas** yang ingin diprediksi.\n\n"
                "Contoh:\n"
                "- *\"Prediksi harga beras di Jakarta\"*\n"
                "- *\"Forecast harga cabai 30 hari\"*\n"
                "- *\"Perkiraan harga bawang merah minggu depan\"*"
            )

        province = provinces[0] if provinces else None

        # Extract forecast horizon from message
        periods = self._extract_forecast_days(msg)

        try:
            from models.prophet_forecast import FoodPriceProphet
        except ImportError:
            return self._response(
                "⚠️ Modul Prophet tidak tersedia. Pastikan library `prophet` terinstall."
            )

        # Load data
        all_data = self.store.load_all()
        if all_data.empty:
            return self._response("⚠️ Tidak ada data untuk melakukan prediksi.")

        loc_text = f" di {province}" if province else " (Nasional)"

        try:
            if province:
                # Province-specific forecast
                subset = all_data[
                    (all_data["province"] == province)
                    & (all_data["commodity"] == commodity)
                ]
                if subset.empty or len(subset) < 30:
                    return self._response(
                        f"⚠️ Data **{commodity}**{loc_text} tidak cukup untuk prediksi "
                        f"(minimal 30 data point, tersedia {len(subset)})."
                    )
                forecaster = FoodPriceProphet(all_data)
                forecast = forecaster.train_and_forecast(
                    province, commodity, periods=periods
                )
            else:
                # National average — pick province with most data
                prov_counts = all_data[
                    all_data["commodity"] == commodity
                ].groupby("province").size()
                if prov_counts.empty:
                    return self._response(
                        f"⚠️ Tidak ditemukan data **{commodity}** di database."
                    )
                best_prov = prov_counts.idxmax()
                province = best_prov
                loc_text = f" di {province} (provinsi referensi)"
                forecaster = FoodPriceProphet(all_data)
                forecast = forecaster.train_and_forecast(
                    province, commodity, periods=periods
                )

            # Get last actual data for comparison
            series = self.store.get_series(province, commodity)
            last_actual = series["price"].iloc[-1] if not series.empty else None
            last_date = series["date"].iloc[-1] if not series.empty else None

            # Extract forecast data (future only)
            if last_date is not None:
                future_forecast = forecast[forecast["ds"] > last_date].copy()
            else:
                future_forecast = forecast.tail(periods).copy()

            if future_forecast.empty:
                return self._response(
                    f"⚠️ Tidak dapat menghasilkan prediksi untuk **{commodity}**{loc_text}."
                )

            # Key prediction points
            pred_7d = future_forecast.iloc[min(6, len(future_forecast) - 1)]
            pred_14d = future_forecast.iloc[min(13, len(future_forecast) - 1)]
            pred_30d = future_forecast.iloc[min(29, len(future_forecast) - 1)]
            pred_last = future_forecast.iloc[-1]

            # Build response
            change_text = ""
            if last_actual is not None:
                change_30 = pred_30d["yhat"] - last_actual
                pct_30 = (change_30 / last_actual) * 100
                if change_30 > 0:
                    change_text = f"\n📈 Harga diprediksi **naik {self._fmt_price(abs(change_30))}** (+{pct_30:.1f}%) dalam 30 hari."
                elif change_30 < 0:
                    change_text = f"\n📉 Harga diprediksi **turun {self._fmt_price(abs(change_30))}** ({pct_30:.1f}%) dalam 30 hari."
                else:
                    change_text = "\n➡️ Harga diprediksi **stabil** dalam 30 hari."

            # Confidence level based on interval width
            avg_range = (future_forecast["yhat_upper"] - future_forecast["yhat_lower"]).mean()
            avg_price = future_forecast["yhat"].mean()
            confidence_pct = max(0, 100 - (avg_range / avg_price * 100))
            conf_label = "Tinggi ✅" if confidence_pct > 80 else "Sedang 🟡" if confidence_pct > 60 else "Rendah ⚠️"

            result_text = (
                f"## 🔮 Prediksi Harga {commodity}{loc_text}\n\n"
                f"Model: **Prophet AI** | Horizon: **{periods} hari**\n\n"
            )

            if last_actual is not None:
                result_text += f"### 📌 Harga Saat Ini\n- **{self._fmt_price(last_actual)}** (per {last_date.strftime('%d %B %Y')})\n\n"

            result_text += (
                f"### 📊 Prediksi Harga\n"
                f"| Periode | Prediksi | Batas Bawah | Batas Atas |\n"
                f"|---------|----------|-------------|------------|\n"
                f"| 7 hari | {self._fmt_price(pred_7d['yhat'])} | {self._fmt_price(pred_7d['yhat_lower'])} | {self._fmt_price(pred_7d['yhat_upper'])} |\n"
                f"| 14 hari | {self._fmt_price(pred_14d['yhat'])} | {self._fmt_price(pred_14d['yhat_lower'])} | {self._fmt_price(pred_14d['yhat_upper'])} |\n"
                f"| 30 hari | {self._fmt_price(pred_30d['yhat'])} | {self._fmt_price(pred_30d['yhat_lower'])} | {self._fmt_price(pred_30d['yhat_upper'])} |\n"
            )

            if len(future_forecast) > 30:
                result_text += (
                    f"| {periods} hari | {self._fmt_price(pred_last['yhat'])} | "
                    f"{self._fmt_price(pred_last['yhat_lower'])} | {self._fmt_price(pred_last['yhat_upper'])} |\n"
                )

            result_text += (
                f"\n### 📈 Ringkasan\n"
                f"- Tingkat kepercayaan: **{conf_label}** ({confidence_pct:.0f}%)\n"
                f"{change_text}\n\n"
                f"> ⚠️ *Prediksi ini menggunakan model Prophet. Hasil aktual dapat berbeda.*"
            )

            # Prepare chart data — combine historical + forecast
            chart_df = pd.DataFrame({
                "date": future_forecast["ds"],
                "price": future_forecast["yhat"],
                "lower": future_forecast["yhat_lower"],
                "upper": future_forecast["yhat_upper"],
            })

            # Prepend last 30 days of actual data
            if not series.empty:
                hist = series.tail(30).copy()
                hist["lower"] = hist["price"]
                hist["upper"] = hist["price"]
                chart_df = pd.concat([hist, chart_df], ignore_index=True)

            return self._response(
                result_text,
                chart_data=chart_df,
                chart_type="forecast",
                chart_title=f"Prediksi Harga {commodity}{loc_text} ({periods} hari)",
            )

        except Exception as e:
            logger.error(f"Forecast error: {e}", exc_info=True)
            return self._response(
                f"⚠️ Terjadi error saat melakukan prediksi: `{e}`\n\n"
                f"Pastikan data **{commodity}**{loc_text} tersedia dan cukup."
            )

    @staticmethod
    def _extract_forecast_days(msg: str) -> int:
        """Extract forecast horizon (days) from user message."""
        msg_lower = msg.lower()

        # Try to find explicit number of days
        m = re.search(r"(\d+)\s*hari", msg_lower)
        if m:
            return min(int(m.group(1)), 120)

        # Week-based
        m = re.search(r"(\d+)\s*minggu", msg_lower)
        if m:
            return min(int(m.group(1)) * 7, 120)

        # Month-based
        m = re.search(r"(\d+)\s*bulan", msg_lower)
        if m:
            return min(int(m.group(1)) * 30, 120)

        # Keywords
        if any(w in msg_lower for w in ["besok", "esok"]):
            return 1
        if "minggu depan" in msg_lower or "pekan depan" in msg_lower:
            return 7
        if "bulan depan" in msg_lower:
            return 30
        if any(w in msg_lower for w in ["3 bulan", "kuartal", "triwulan"]):
            return 90

        # Default: 30 days
        return 30

    def _handle_unknown(self, msg, commodity, provinces):
        # If commodity or province was detected, try price lookup anyway
        if commodity or provinces:
            return self._handle_harga_terbaru(msg, commodity, provinces)

        return self._response(
            "🤔 Maaf, saya belum memahami pertanyaan Anda.\n\n"
            "Coba ajukan pertanyaan seperti:\n"
            "- *\"Berapa harga beras di Jakarta?\"*\n"
            "- *\"Prediksi harga cabai di Jakarta\"*\n"
            "- *\"Tren harga cabai merah\"*\n"
            "- *\"Bandingkan harga bawang merah Jakarta dan Surabaya\"*\n"
            "- *\"Statistik harga gula pasir\"*\n\n"
            "Ketik **\"help\"** untuk melihat panduan lengkap."
        )
