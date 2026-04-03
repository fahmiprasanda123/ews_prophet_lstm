import pandas as pd
import numpy as np

def harmonize_data(input_file="food_prices_raw_scraped.csv", output_file="food_prices_real.csv"):
    """Clean, interpolate and finalize PIHPS data."""
    if not pd.io.common.file_exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    print(f"🧹 Harmonizing {input_file}...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop duplicates just in case
    df = df.drop_duplicates(subset=['date', 'province', 'commodity'])
    
    # We will iterate through each province and commodity
    results = []
    
    provinces = df['province'].unique()
    commodities = df['commodity'].unique()
    
    # Create the target date range (Daily)
    full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    
    for prov in provinces:
        for com in commodities:
            subset = df[(df['province'] == prov) & (df['commodity'] == com)].copy()
            if subset.empty:
                continue
                
            # Reindex to full date range to find gaps (PIHPS skips weekends/holidays)
            subset = subset.set_index('date').reindex(full_date_range)
            
            # Fill constant columns
            subset['province'] = prov
            subset['commodity'] = com
            
            # Interpolate price (Linear)
            # Limit = 7 ensures we don't interpolate over huge gaps (e.g. data missing for a month)
            subset['price'] = subset['price'].interpolate(method='linear', limit_direction='both', limit=7)
            
            # Final forward/back fill for edges if needed
            subset['price'] = subset['price'].ffill().bfill()
            
            subset = subset.reset_index().rename(columns={'index': 'date'})
            results.append(subset)
            
    df_final = pd.concat(results)
    
    # Ensure correct format for models
    df_final = df_final[['date', 'province', 'commodity', 'price']]
    df_final.to_csv(output_file, index=False)
    print(f"✅ Success! Harmonized data saved to {output_file} with {len(df_final)} rows.")

if __name__ == "__main__":
    harmonize_data()
