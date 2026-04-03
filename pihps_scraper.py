import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

class PihpsScraper:
    def __init__(self):
        self.base_url = "https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah"
        # Standard Mapping from PIHPS BI
        self.commodity_map = {
            'Beras': 'Beras',
            'Daging Ayam Ras Segar': 'Daging Ayam',
            'Daging Sapi Kualitas 1': 'Daging Sapi',
            'Telur Ayam Ras Segar': 'Telur Ayam',
            'Bawang Merah Ukuran Sedang': 'Bawang Merah',
            'Bawang Putih Ukuran Sedang': 'Bawang Putih',
            'Cabai Merah Besar': 'Cabai Merah',
            'Cabai Rawit Merah': 'Cabai Rawit',
            'Minyak Goreng Curah': 'Minyak Goreng',
            'Gula Pasir Lokal': 'Gula Pasir'
        }
        
    def fetch_data(self, province_id, start_date, end_date, retries=3):
        """Fetch data with retry logic and increased timeout."""
        params = {
            'price_type_id': 1,
            'tipe_laporan': 1,
            'start_date': start_date,
            'end_date': end_date,
            'province_id': province_id
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalDaerah'
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"  ⚠️ Attempt {attempt+1} failed for prov {province_id}: {e}")
                time.sleep(2 * (attempt + 1)) # Exponential backoff
        return None

    def process_json(self, json_data, province_name):
        """Convert PIHPS JSON to tidy format."""
        if not json_data or 'data' not in json_data:
            return []
        
        tidy_data = []
        for item in json_data['data']:
            orig_name = item.get('name', '')
            if orig_name in self.commodity_map:
                std_commodity = self.commodity_map[orig_name]
                
                # Iterate over keys that look like dates (dd/mm/yyyy)
                for key, val in item.items():
                    if '/' in key and len(key) == 10:
                        try:
                            # Clean price (PIHPS uses '12.000' or '12,000' depending on locale, but usually dot for thousands)
                            # Actually from test_data.json it's "15,900" (comma as thousands)
                            price_str = val.replace(',', '').replace('.', '')
                            if price_str.isdigit():
                                price = float(price_str)
                                # PIHPS dates are dd/mm/yyyy
                                dt = datetime.strptime(key, '%d/%m/%Y').strftime('%Y-%m-%d')
                                tidy_data.append({
                                    'date': dt,
                                    'province': province_name,
                                    'commodity': std_commodity,
                                    'price': price
                                })
                        except:
                            continue
        return tidy_data

def get_province_list():
    """Load province list from generated reference."""
    with open('province_ref.json', 'r') as f:
        return json.load(f)['data']

if __name__ == "__main__":
    scraper = PihpsScraper()
    provinces = get_province_list()
    
    # Example: Fetching last 30 days for 1 province to test
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Testing scraper for {provinces[0]['name']}...")
    raw = scraper.fetch_data(provinces[0]['id'], start, end)
    tidy = scraper.process_json(raw, provinces[0]['name'])
    
    df = pd.DataFrame(tidy)
    print(df.head())
