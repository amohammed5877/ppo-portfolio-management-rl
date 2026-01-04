# data_loader.py

import pandas as pd

def load_flat_stock_data(filepath="stocks_5y.csv", tickers=None):
    """
    Loads flat CSV data with columns like MSFT_Open, MSFT_High, etc.
    Converts it to long format with columns: Date, Open, High, Low, Close, Volume, ticker
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    long_frames = []

    for ticker in tickers:
        try:
            temp_df = df[["Date", 
                          f"{ticker}_Open", 
                          f"{ticker}_High", 
                          f"{ticker}_Low", 
                          f"{ticker}_Close", 
                          f"{ticker}_Volume"]].copy()
            temp_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            temp_df["ticker"] = ticker
            long_frames.append(temp_df)
        except KeyError:
            print(f"⚠️ Missing data for {ticker}, skipping...")

    if not long_frames:
        raise ValueError("❌ No valid ticker data found in DataFrame.")

    return pd.concat(long_frames).sort_values(by=["Date", "ticker"]).reset_index(drop=True)
