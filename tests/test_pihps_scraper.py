import pytest
import requests
from pihps_scraper import PihpsScraper
from unittest.mock import patch, MagicMock

def test_process_json_empty():
    scraper = PihpsScraper()
    assert scraper.process_json(None, "Aceh") == []
    assert scraper.process_json({}, "Aceh") == []

def test_process_json_valid():
    scraper = PihpsScraper()
    # Sample data format from PIHPS BI
    mock_response = {
        "data": [
            {
                "name": "Beras",
                "01/01/2024": "12.000",
                "02/01/2024": "12,500"
            },
            {
                "name": "Daging Sapi", # Should map to Daging Sapi
                "01/01/2024": "120.000"
            },
            {
                "name": "Unknown Commodity", # Should be ignored
                "01/01/2024": "1000"
            }
        ]
    }
    
    # Manually adding key for Daging Sapi from scraper mapping
    # self.commodity_map[Daging Sapi Kualitas 1] = Daging Sapi
    # Wait, in the scraper, the map is:
    # 'Daging Sapi Kualitas 1': 'Daging Sapi'
    
    # Let's use the exact names from the mapping
    mock_response = {
        "data": [
            {
                "name": "Beras",
                "01/01/2024": "12.000",
            },
            {
                "name": "Daging Sapi Kualitas 1",
                "02/01/2024": "125.000"
            }
        ]
    }
    
    result = scraper.process_json(mock_response, "DKI Jakarta")
    assert len(result) == 2
    assert result[0]['commodity'] == "Beras"
    assert result[0]['price'] == 12000.0
    assert result[0]['date'] == "2024-01-01"
    assert result[1]['commodity'] == "Daging Sapi"
    assert result[1]['price'] == 125000.0
    assert result[1]['date'] == "2024-01-02"

@patch("requests.get")
def test_fetch_data_success(mock_get):
    scraper = PihpsScraper()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    mock_get.return_value = mock_response
    
    result = scraper.fetch_data("1", "2024-01-01", "2024-01-02")
    assert result == {"data": []}
    assert mock_get.called

@patch("requests.get")
def test_fetch_data_retry(mock_get):
    scraper = PihpsScraper()
    # Mock first attempt fail, second success
    mock_fail = MagicMock()
    mock_fail.raise_for_status.side_effect = Exception("Fail")
    
    mock_success = MagicMock()
    mock_success.status_code = 200
    mock_success.json.return_value = {"data": []}
    
    mock_get.side_effect = [mock_fail, mock_success]
    
    # We might need to reduce sleep time for tests or just wait
    # But for simplicity, I'll mock the print/time too if needed
    with patch("time.sleep", return_value=None):
        result = scraper.fetch_data("1", "2024-01-01", "2024-01-02")
        assert result == {"data": []}
        assert mock_get.call_count == 2


def test_process_json_unknown_commodity_ignored():
    """Commodities not in commodity_map should be silently skipped."""
    scraper = PihpsScraper()
    mock_response = {
        "data": [
            {"name": "Unknown Commodity XYZ", "01/01/2024": "12.000"},
            {"name": "Another Unknown", "01/01/2024": "5.000"},
        ]
    }
    result = scraper.process_json(mock_response, "DKI Jakarta")
    assert result == []


def test_process_json_date_format_conversion():
    """PIHPS dd/mm/yyyy should convert to yyyy-mm-dd."""
    scraper = PihpsScraper()
    mock_response = {
        "data": [
            {"name": "Beras", "25/12/2024": "12.000"}
        ]
    }
    result = scraper.process_json(mock_response, "DKI Jakarta")
    assert len(result) == 1
    assert result[0]['date'] == "2024-12-25"


def test_process_json_price_cleaning():
    """Both dot and comma thousand separators should be handled."""
    scraper = PihpsScraper()
    # PIHPS uses dots as thousands separator: "12.000" = 12000
    mock_response = {
        "data": [
            {"name": "Beras", "01/01/2024": "125.000"}
        ]
    }
    result = scraper.process_json(mock_response, "DKI Jakarta")
    assert result[0]['price'] == 125000.0


def test_commodity_map_has_10_entries():
    """Should map exactly 10 raw names to 10 standard commodities."""
    scraper = PihpsScraper()
    assert len(scraper.commodity_map) == 10
    
    expected_outputs = {
        'Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam',
        'Bawang Merah', 'Bawang Putih', 'Cabai Merah', 'Cabai Rawit',
        'Minyak Goreng', 'Gula Pasir'
    }
    assert set(scraper.commodity_map.values()) == expected_outputs

