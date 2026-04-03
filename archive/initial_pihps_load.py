from pihps_scraper import PihpsScraper, get_province_list
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def run_initial_load():
    scraper = PihpsScraper()
    provinces = get_province_list()
    
    # 5-Year Range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2021-01-01"
    
    # 6-Month batches to avoid server timeouts
    date_batches = [
        ("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
        ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
        ("2026-01-01", datetime.now().strftime('%Y-%m-%d'))
    ]
    
    all_data = []
    
    print(f"🚀 Starting Initial PIHPS Load (5 Years)...")
    print(f"Target: {len(provinces)} provinces.")
    
    for prov in provinces:
        print(f"\n--- Fetching: {prov['name']} ---")
        for start_sync, end_sync in date_batches:
            if datetime.strptime(start_sync, '%Y-%m-%d') > datetime.now():
                continue
                
            print(f"  Fetching {start_sync} to {end_sync}...", end=" ", flush=True)
            raw = scraper.fetch_data(prov['id'], start_sync, end_sync)
            if raw:
                tidy = scraper.process_json(raw, prov['name'])
                all_data.extend(tidy)
                print(f"Done ({len(tidy)} records)")
            else:
                print("Failed after retries")
            
            # Rate limiting
            time.sleep(1.0)
            
        # Partial save to avoid data loss
        if len(all_data) > 0:
            df_temp = pd.DataFrame(all_data)
            df_temp.to_csv("food_prices_raw_scraped.csv", index=False)
            
    print(f"\n✅ Initial Load Complete! Total records: {len(all_data)}")
    df_final = pd.DataFrame(all_data)
    df_final.to_csv("food_prices_raw_scraped.csv", index=False)
    return df_final

if __name__ == "__main__":
    run_initial_load()
