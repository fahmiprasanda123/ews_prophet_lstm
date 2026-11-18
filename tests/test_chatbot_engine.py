"""Tests untuk ChatbotEngine: pencocokan entitas, intent, horizon prediksi, rata-rata nasional."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from engine.chatbot_engine import ChatbotEngine

PROVINCES = ["Bali", "DKI Jakarta", "Jawa Barat", "Kepulauan Riau", "Maluku", "Maluku Utara",
             "Papua", "Papua Barat", "Riau"]
COMMODITIES = ["Beras", "Cabai Merah", "Cabai Rawit"]


class FakeStore:
    def __init__(self, df):
        self.df = df

    def get_provinces(self):
        return sorted(self.df["province"].unique())

    def get_commodities(self):
        return sorted(self.df["commodity"].unique())

    def load_all(self):
        return self.df.copy()

    def get_series(self, province, commodity):
        s = self.df[(self.df["province"] == province) & (self.df["commodity"] == commodity)]
        return s[["date", "price"]].sort_values("date").reset_index(drop=True)

    def get_latest_prices(self, commodity=None):
        d = self.df if commodity is None else self.df[self.df["commodity"] == commodity]
        idx = d.groupby(["province", "commodity"])["date"].idxmax()
        return d.loc[idx].sort_values("province").reset_index(drop=True)


def make_df(days=120):
    dates = pd.bdate_range("2026-06-01", periods=days)
    rows = []
    for i, prov in enumerate(PROVINCES):
        for comm in COMMODITIES:
            for d in dates:
                rows.append({"date": d, "province": prov, "commodity": comm, "price": 10000.0 + 1000 * i})
    return pd.DataFrame(rows)


def make_bot(df=None):
    bot = ChatbotEngine.__new__(ChatbotEngine)
    bot.store = FakeStore(df if df is not None else make_df())
    bot._provinces = None
    bot._commodities = None
    return bot


class TestEntityExtraction:
    def setup_method(self):
        self.bot = make_bot()

    @pytest.mark.parametrize("msg,expected", [
        ("Bandingkan beras Papua Barat dan Bali", ["Papua Barat", "Bali"]),
        ("harga cabai di Maluku Utara", ["Maluku Utara"]),
        ("harga beras Kepulauan Riau", ["Kepulauan Riau"]),
        ("saya kembali lagi", []),
        ("bandingkan beras jawa barat vs jakarta", ["Jawa Barat", "DKI Jakarta"]),
        ("harga beras di Riau dan Papua", ["Riau", "Papua"]),
    ])
    def test_provinces(self, msg, expected):
        assert self.bot._extract_provinces(msg) == expected

    def test_commodity_word_boundary(self):
        assert self.bot._extract_commodity("berasal dari mana data ini?") is None
        assert self.bot._extract_commodity("harga cabai rawit") == "Cabai Rawit"
        assert self.bot._extract_commodity("harga cabai") == "Cabai Merah"

    def test_greeting_does_not_override_question(self):
        assert self.bot._detect_intent("halo") == "salam"
        assert self.bot._detect_intent("halo, berapa harga beras di Jakarta?") == "harga_terbaru"
        r = self.bot.process("halo, berapa harga beras di Jakarta?")
        assert "Harga Beras di DKI Jakarta" in r["text"]
        assert r["chart_type"] == "line"


class TestTrendAndNational:
    def test_single_row_series_no_nameerror(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2026-11-18")], "province": ["Bali"],
                           "commodity": ["Beras"], "price": [12000.0]})
        r = make_bot(df).process("tren harga beras di Bali")
        assert "Rp 12.000" in r["text"]

    def test_national_trend_ignores_missing_province_day(self):
        df = make_df()
        last = df["date"].max()
        # Provinsi termahal tidak melapor di hari terakhir: rata-rata naif turun, versi seimbang tidak
        df = df[~((df["province"] == "Riau") & (df["date"] == last))]
        r = make_bot(df).process("tren harga beras")
        assert "+0.0%" in r["text"]

    def test_latest_overview_excludes_stale(self):
        df = make_df()
        cutoff = df["date"].max() - pd.Timedelta(days=60)
        df = df[~((df["province"] == "Papua") & (df["commodity"] == "Beras") & (df["date"] > cutoff))]
        r = make_bot(df).process("harga beras")
        assert "Papua (terakhir" in r["text"]
        assert "Data per" in r["text"]


class TestForecastHorizon:
    @pytest.fixture
    def bot(self, monkeypatch):
        class FakeProphet:
            def __init__(self, df):
                self.df = df

            def train_and_forecast(self, province, commodity, periods=30):
                last = self.df["date"].max()
                ds = pd.date_range(last + pd.Timedelta(days=1), periods=periods)
                y = 10000.0 + 10 * np.arange(1, periods + 1)
                return pd.DataFrame({"ds": ds, "yhat": y, "yhat_lower": y - 50, "yhat_upper": y + 50})

        mod = types.ModuleType("models.prophet_forecast")
        mod.FoodPriceProphet = FakeProphet
        monkeypatch.setitem(sys.modules, "models.prophet_forecast", mod)
        return make_bot()

    def test_short_horizon_only_shows_requested_rows(self, bot):
        r = bot.process("Prediksi harga beras di Bali 7 hari")
        assert "| 7 hari |" in r["text"]
        assert "| 14 hari |" not in r["text"] and "| 30 hari |" not in r["text"]
        assert "dalam 7 hari" in r["text"] and "dalam 30 hari" not in r["text"]

    def test_long_horizon_rows(self, bot):
        r = bot.process("Prediksi harga beras di Bali 60 hari")
        for h in (7, 14, 30, 60):
            assert f"| {h} hari |" in r["text"]
        assert "dalam 60 hari" in r["text"]

    def test_horizon_points(self):
        assert ChatbotEngine._horizon_points(7) == [7]
        assert ChatbotEngine._horizon_points(10) == [7, 10]
        assert ChatbotEngine._horizon_points(30) == [7, 14, 30]
        assert ChatbotEngine._horizon_points(1) == [1]
