import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os

# === SETTINGS ===
symbol = "BTCUSDT"
interval = "1m"
start_date = datetime(2015, 1, 1, tzinfo=timezone.utc)  # 5 years ago
end_date = datetime.now(timezone.utc)
save_file = "ohlcv_20150101.csv"


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
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


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
current = get_last_saved_timestamp(save_file) or start_date

while current < end_date:
    try:
        print(f"Fetching from {current.isoformat()} ...")
        df = get_binance_ohlcv(symbol, interval, current)
        if df.empty:
            print("No data returned. Sleeping 5s...")
            time.sleep(5)
            continue

        # Avoid duplicates
        if os.path.exists(save_file):
            existing = pd.read_csv(save_file, usecols=["timestamp"])
            df = df[~df["timestamp"].isin(existing["timestamp"])]

        # Append to CSV
        df.to_csv(
            save_file, mode="a", header=not os.path.exists(save_file), index=False
        )

        # Move forward
        last_ts = int(df["timestamp"].iloc[-1])
        current = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(
            minutes=1
        )
        time.sleep(0.2)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)
        continue

print("✅ Done! BTC 1-minute OHLCV data saved to btc_1min_full.csv.")
