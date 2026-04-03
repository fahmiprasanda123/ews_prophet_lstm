import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data():
    # Setup
    start_date = datetime(2021, 1, 1)
    end_date = datetime.now() # Generates up to today (April 2026)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    provinces = [
        'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Kepulauan Riau', 
        'Jambi', 'Sumatera Selatan', 'Kepulauan Bangka Belitung', 'Bengkulu', 'Lampung', 
        'DKI Jakarta', 'Jawa Barat', 'Banten', 'Jawa Tengah', 'DI Yogyakarta', 
        'Jawa Timur', 'Bali', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 
        'Kalimantan Barat', 'Kalimantan Tengah', 'Kalimantan Selatan', 'Kalimantan Timur', 
        'Kalimantan Utara', 'Sulawesi Utara', 'Gorontalo', 'Sulawesi Tengah', 
        'Sulawesi Barat', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Maluku Utara', 
        'Maluku', 'Papua Barat', 'Papua', 'Papua Tengah', 'Papua Selatan', 
        'Papua Pegunungan', 'Papua Barat Daya'
    ]
    
    commodities = [
        'Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam', 'Bawang Merah', 
        'Bawang Putih', 'Cabai Merah', 'Cabai Rawit', 'Minyak Goreng', 'Gula Pasir'
    ]
    
    # Base prices (IDR/kg or similar) and volatility
    commodity_config = {
        'Beras': {'base': 12000, 'vol': 0.005, 'seasonality': 0.05},
        'Daging Ayam': {'base': 35000, 'vol': 0.02, 'seasonality': 0.15},
        'Daging Sapi': {'base': 120000, 'vol': 0.01, 'seasonality': 0.20},
        'Telur Ayam': {'base': 28000, 'vol': 0.015, 'seasonality': 0.10},
        'Bawang Merah': {'base': 30000, 'vol': 0.04, 'seasonality': 0.30},
        'Bawang Putih': {'base': 25000, 'vol': 0.03, 'seasonality': 0.10},
        'Cabai Merah': {'base': 45000, 'vol': 0.06, 'seasonality': 0.40},
        'Cabai Rawit': {'base': 60000, 'vol': 0.08, 'seasonality': 0.50},
        'Minyak Goreng': {'base': 18000, 'vol': 0.02, 'seasonality': 0.05},
        'Gula Pasir': {'base': 15000, 'vol': 0.01, 'seasonality': 0.05}
    }
    
    # Regional price multipliers (e.g., Eastern Indonesia more expensive)
    province_multipliers = {p: 1.0 for p in provinces}
    eastern_provinces = [
        'Maluku Utara', 'Maluku', 'Papua Barat', 'Papua', 'Papua Tengah', 
        'Papua Selatan', 'Highland Papua', 'Papua Barat Daya'
    ]
    for p in eastern_provinces:
        if p in province_multipliers:
            province_multipliers[p] = 1.3 # 30% more expensive
            
    data = []
    
    print(f"Generating data for {len(provinces)} provinces and {len(commodities)} commodities...")
    
    for province in provinces:
        p_mult = province_multipliers.get(province, 1.0)
        for commodity in commodities:
            cfg = commodity_config[commodity]
            base_p = cfg['base'] * p_mult
            vol = cfg['vol']
            seas_amp = cfg['seasonality']
            
            # Generate time series
            underlying_prices = [base_p]
            data_points = []
            
            for i in range(len(date_range)):
                # Trend (inflation)
                trend = 1.0001 ** i # base_p * (1.0001^i)
                
                # Seasonality (Annual)
                day_of_year = date_range[i].dayofyear
                seasonality = 1 + seas_amp * np.sin(2 * np.pi * day_of_year / 365)
                
                # Major holiday shocks (e.g., Eid Al-Fitr spike)
                holiday_spike = 1.0
                if 100 < day_of_year < 120:
                    holiday_spike = 1.1 + np.random.uniform(0, 0.1)
                
                # Random shock (EWS trigger)
                shock = 1.0
                if np.random.random() < 0.005: 
                    shock = 1.2 + np.random.uniform(0, 0.3)
                    
                # Walk part (accumulated noise)
                if i == 0:
                    noise_factor = 1.0
                else:
                    noise_factor = underlying_prices[-1] / base_p * (1 + np.random.normal(0, vol))
                
                current_underlying = base_p * noise_factor
                underlying_prices.append(current_underlying)
                
                # Apply trend, seasonality and spikes to the underlying price
                final_price = current_underlying * trend * seasonality * holiday_spike * shock
                data_points.append([date_range[i], province, commodity, round(final_price, 2)])
                
            # Append to master data
            data.extend(data_points)
                
    df = pd.DataFrame(data, columns=['date', 'province', 'commodity', 'price'])
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("food_prices_indonesia.csv", index=False)
    print(f"Dataset saved to food_prices_indonesia.csv with {len(df)} rows.")
