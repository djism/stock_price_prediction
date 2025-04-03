import yfinance as yf
import pandas as pd
import os
from config import TICKER, START_DATE, END_DATE, DATA_DIR

def fetch_stock_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    data = yf.download(TICKER, start=START_DATE, end=END_DATE)
    file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    data.to_csv(file_path)
    print(f"Data saved at {file_path}")

if __name__ == "__main__":
    fetch_stock_data()
