import pytest
import pandas as pd
from ews_engine import EWSEngine

def test_calculate_warning_level_danger():
    ews = EWSEngine(pd.DataFrame())
    # 25% increase -> Danger
    level, msg = ews.calculate_warning_level(10000, 12500)
    assert level == "Danger"
    assert "High risk" in msg

def test_calculate_warning_level_alert():
    ews = EWSEngine(pd.DataFrame())
    # 15% increase -> Alert
    level, msg = ews.calculate_warning_level(10000, 11500)
    assert level == "Alert"
    assert "Moderate" in msg

def test_calculate_warning_level_normal():
    ews = EWSEngine(pd.DataFrame())
    # 5% increase -> Normal
    level, msg = ews.calculate_warning_level(10000, 10500)
    assert level == "Normal"
    assert "stable" in msg

def test_calculate_warning_level_downward():
    ews = EWSEngine(pd.DataFrame())
    # 15% decrease -> Normal (Downward)
    level, msg = ews.calculate_warning_level(10000, 8500)
    assert level == "Normal (Downward)"
    assert "decrease" in msg

def test_get_status_color():
    ews = EWSEngine(pd.DataFrame())
    assert ews.get_status_color("Danger") == "#FF4B4B"
    assert ews.get_status_color("Alert") == "#FFA500"
    assert ews.get_status_color("Normal") == "#00CC96"
    assert ews.get_status_color("Unknown") == "#FFFFFF"
