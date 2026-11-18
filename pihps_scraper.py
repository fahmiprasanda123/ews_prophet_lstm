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
                        price = parse_price(val)
                        if price is None:
                            continue
                        try:
                            dt = datetime.strptime(key, '%d/%m/%Y').strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                        tidy_data.append({
                            'date': dt,
                            'province': province_name,
                            'commodity': std_commodity,
                            'price': price
                        })
        return tidy_data


def parse_price(val):
    """Ubah nilai harga PIHPS menjadi float; None bila kosong, tidak valid, atau <= 0.

    Format Indonesia: '.' pemisah ribuan, ',' desimal ("12.500,50" = 12500.5).
    PIHPS kadang juga mengirim "15,900" (koma sebagai ribuan). Bila hanya satu jenis
    pemisah muncul sekali dan diikuti tepat 3 digit, pemisah itu dianggap ribuan.
    """
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        price = float(val)
    else:
        s = str(val).strip().replace(' ', '')
        if not s or not any(ch.isdigit() for ch in s):
            return None
        if '.' in s and ',' in s:
            dec = '.' if s.rfind('.') > s.rfind(',') else ','
            thou = ',' if dec == '.' else '.'
            s = s.replace(thou, '').replace(dec, '.')
        elif '.' in s or ',' in s:
            sep = '.' if '.' in s else ','
            parts = s.split(sep)
            if len(parts) > 2 or len(parts[-1]) == 3:
                s = s.replace(sep, '')
            else:
                s = s.replace(sep, '.')
        try:
            price = float(s)
        except ValueError:
            return None
    if price != price or price <= 0 or price == float('inf'):
        return None
    return price


PROVINCE_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'province_ref.json')


def get_province_list():
    """Load province list from generated reference."""
    with open(PROVINCE_REF, 'r') as f:
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
