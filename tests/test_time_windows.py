"""Tests untuk helper jendela waktu kalender, NaN guard EWS, dan pembuatan PDF."""
import numpy as np
import pandas as pd

from engine.time_windows import (
    balanced_national_series, last_n_days, pct_change_days, price_days_ago, split_stale,
)


def business_series(days=60, start="2026-01-01", step=100.0):
    """Seri hari kerja saja (meniru PIHPS), harga naik 'step' per baris."""
    dates = pd.bdate_range(start, periods=days)
    return pd.DataFrame({"date": dates, "price": 10000 + step * np.arange(days)})


class TestCalendarWindows:
    def test_price_days_ago_uses_calendar_not_rows(self):
        s = business_series()
        last = s["date"].iloc[-1]
        expected = s[s["date"] <= last - pd.Timedelta(days=7)]["price"].iloc[-1]
        assert price_days_ago(s, 7) == expected
        # 7 hari kalender = 5 hari kerja, bukan 7 baris (iloc[-7] lama = 6 baris ke belakang)
        assert price_days_ago(s, 7) == s["price"].iloc[-6]

    def test_price_days_ago_none_when_too_short(self):
        s = business_series(days=3)
        assert price_days_ago(s, 30) is None
        assert pct_change_days(s, 30) is None

    def test_last_n_days_window(self):
        s = business_series(days=100)
        w = last_n_days(s, 30)
        assert (w["date"].iloc[-1] - w["date"].iloc[0]).days <= 30
        assert len(w) < 30  # akhir pekan tidak ada, jadi < 30 baris

    def test_split_stale(self):
        latest = pd.DataFrame({
            "date": pd.to_datetime(["2026-11-18", "2026-11-17", "2026-04-03"]),
            "province": ["A", "B", "C"], "price": [1.0, 2.0, 3.0],
        })
        fresh, stale = split_stale(latest)
        assert list(fresh["province"]) == ["A", "B"]
        assert list(stale["province"]) == ["C"]


class TestBalancedNational:
    def test_missing_province_does_not_create_fake_swing(self):
        dates = pd.bdate_range("2026-01-01", periods=10)
        rows = []
        for d in dates:
            rows.append({"date": d, "province": "Murah", "price": 10000.0})
            # Provinsi mahal absen pada hari ke-5 -> rata-rata naif melonjak turun
            if d != dates[5]:
                rows.append({"date": d, "province": "Mahal", "price": 20000.0})
        df = pd.DataFrame(rows)
        naive = df.groupby("date")["price"].mean()
        assert naive.min() == 10000.0  # bukti masalah lama
        bal = balanced_national_series(df)
        assert bal["price"].nunique() == 1
        assert bal["price"].iloc[0] == 15000.0

    def test_stale_province_excluded(self):
        dates = pd.bdate_range("2026-01-01", periods=40)
        rows = [{"date": d, "province": "Aktif", "price": 10000.0} for d in dates]
        rows += [{"date": d, "province": "Berhenti", "price": 50000.0} for d in dates[:10]]
        bal = balanced_national_series(pd.DataFrame(rows))
        assert (bal["price"] == 10000.0).all()
        assert bal["date"].iloc[-1] == dates[-1]


class TestEWSNaNGuard:
    def test_cross_region_single_province_not_nan(self):
        from engine.ews_engine_v2 import EWSEngineV2
        dates = pd.bdate_range("2026-01-01", periods=60)
        df = pd.DataFrame({"date": dates, "province": "DKI Jakarta", "commodity": "Beras",
                           "price": 10000.0 + np.arange(60)})
        ews = EWSEngineV2(df)
        score = ews._score_cross_region("Beras", 10050.0, 10100.0)
        assert not np.isnan(score)
        assert score < 100

    def test_velocity_short_series_neutral(self):
        from engine.ews_engine_v2 import EWSEngineV2
        s = business_series(days=10)  # < 30 baris data
        df = s.assign(province="X", commodity="Y")
        assert EWSEngineV2(df)._score_velocity(s, 12000) == 30


class TestHolidayYearBoundary:
    def test_new_year_seen_from_late_december(self):
        from engine.price_narrative import PriceNarrativeAnalyzer
        df = business_series(days=60).assign(province="X", commodity="Beras")
        hol = PriceNarrativeAnalyzer(df)._get_approximate_holidays(2026)
        assert pd.Timestamp("2027-01-01") in hol["natal_tahun_baru"]
        assert pd.Timestamp("2025-12-25") in hol["natal_tahun_baru"]


class TestPdfReport:
    def test_generate_report_with_metrics(self):
        from report.pdf_generator import EWSReportGenerator
        pdf = EWSReportGenerator().generate_report(
            province="DKI Jakarta", commodity="Beras", current_price=15000, predicted_price=15500,
            ews_result={"level": "Watch", "score": 40, "message": "Tren perlu diawasi.",
                        "factors": {"price_change": 30.0}, "recommendations": ["Pantau harga."]},
            supply_risk={"score": 30, "factors": {"trend_momentum": 20.0}, "trend_direction": "Naik",
                         "weekly_change_pct": 1.2, "description": "Risiko pasokan RENDAH."},
            metrics={"RMSE": 100.0, "MAE": 80.0, "MAPE (%)": 2.5, "R²": 0.91},
            forecast_date=pd.Timestamp("2026-12-18"),
        )
        assert bytes(pdf[:4]) == b"%PDF"
