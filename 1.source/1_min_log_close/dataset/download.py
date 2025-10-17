# Download BTC 1m ohlcv dataset
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os

# === SETTINGS ===
symbol = "BTCUSDT"
interval = "1m"
end_date = datetime.now(timezone.utc) # 50 days data
start_date = end_date - timedelta(days=52)
save_file = "Bitcoin.csv"

# === Fetch OHLCV from Binance ===
def get_binance_ohlcv(symbol, interval, start_time, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": int(start_time.timestamp() * 1000),
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw_data = response.json()

    df = pd.DataFrame(
        raw_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["timestamp"] = df["open_time"]  # keep timestamp in ms
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")  # Convert to datetime
    df["datetime"] = df["open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp
    
    return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


# === Get Last Timestamp from File if Exists ===
def get_last_saved_timestamp(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, usecols=["timestamp"])
        if not df.empty:
            last_ts = int(df["timestamp"].iloc[-1])
            return datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(
                minutes=1
            )
    except Exception:
        pass
    return None


# === MAIN DOWNLOAD LOOP ===
# go one folder up
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
save_file_path = os.path.join(parent_dir, "data", save_file)

current = get_last_saved_timestamp(save_file_path) or start_date

while current < end_date:
    try:
        print(f"Fetching from {current.isoformat()} ...")
        df = get_binance_ohlcv(symbol, interval, current)
        if df.empty:
            print("No data returned. Sleeping 1s...")
            time.sleep(1)
            continue

        # Avoid duplicates
        if os.path.exists(save_file_path):
            existing = pd.read_csv(save_file_path, usecols=["timestamp"])
            df = df[~df["timestamp"].isin(existing["timestamp"])]

        # Append to CSV
        df.to_csv(
            save_file_path, mode="a", header=not os.path.exists(save_file_path), index=False
        )

        # Move forward
        last_ts = int(df["timestamp"].iloc[-1])
        current = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(
            minutes=1
        )
        time.sleep(0.2)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        continue

print("Data download completed.")
