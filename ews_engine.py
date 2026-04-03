import pandas as pd
import numpy as np

class EWSEngine:
    def __init__(self, historical_df):
        self.df = historical_df
        
    def calculate_warning_level(self, current_price, predicted_price, threshold_percent=0.10):
        # Base logic: Percent change
        pct_change = (predicted_price - current_price) / current_price
        
        # Thresholds (can be more complex based on volatility)
        if pct_change >= 0.20:
            return "Danger", "⚠️ High risk of acute price surge!"
        elif pct_change >= 0.10:
            return "Alert", "🔸 Moderate price increase predicted."
        elif pct_change <= -0.10:
            return "Normal (Downward)", "📉 Price is expected to decrease."
        else:
            return "Normal", "✅ Price is stable."

    def get_status_color(self, level):
        colors = {
            "Danger": "#FF4B4B", # Red
            "Alert": "#FFA500",  # Orange
            "Normal": "#00CC96", # Green
            "Normal (Downward)": "#00CC96"
        }
        return colors.get(level, "#FFFFFF")

if __name__ == "__main__":
    # Test
    ews = EWSEngine(pd.DataFrame())
    level, msg = ews.calculate_warning_level(10000, 12500)
    print(f"Status: {level} - {msg}")
