import pandas as pd
from models.tft_forecast import get_tft_forecaster

df = pd.read_csv("food_prices_real.csv")
tft = get_tft_forecaster()
if tft.is_available:
    dataset, data = tft.prepare_dataset(df, province="Jawa Timur", commodity="Bawang Merah")
    tft.train(dataset, max_epochs=1, batch_size=32)
    try:
        res = tft.predict(data, dataset)
        print("Success:", res)
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
