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
