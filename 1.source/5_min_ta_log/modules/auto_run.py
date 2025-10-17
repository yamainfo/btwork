import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
import requests

### DB Load part ###
# Download BTC 1m ohlcv dataset
import requests

# === SETTINGS ===
symbol = "BTCUSDT"
interval = "5m"

# === Fetch OHLCV from Binance ===
def get_binance_ohlcv(symbol, interval, start_time, limit=1000):
    # url = "https://api.binance.com/api/v3/klines"
    url = "https://api.binance.us/api/v3/klines"
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


# === Fetch OHLCV from yfinance ===
def get_yfinance_ohlcv(symbol, interval, start_time, limit=1000):
    """
    Fetch OHLCV data from yfinance in the same format as get_binance_ohlcv
    """
    # Convert Binance symbol format to yfinance format
    if symbol == "BTCUSDT":
        yf_symbol = "BTC-USD"
    elif symbol == "ETHUSDT":
        yf_symbol = "ETH-USD"
    else:
        # For other symbols, try to convert USDT to USD
        yf_symbol = symbol.replace("USDT", "-USD")
    
    # Calculate end time (start_time + limit minutes)
    end_time = start_time + timedelta(minutes=limit)
    
    try:
        # Let yfinance handle the session automatically (no custom session)
        ticker = yf.Ticker(yf_symbol)
        
        # Try multiple times with different approaches
        data = None
        for attempt in range(3):
            try:
                if attempt == 0:
                    # First attempt: standard approach
                    data = ticker.history(start=start_time, end=end_time, interval='5m')
                elif attempt == 1:
                    # Second attempt: use period instead of start/end
                    data = ticker.history(period="1d", interval='5m')
                else:
                    # Third attempt: use different symbol format
                    if yf_symbol == "BTC-USD":
                        ticker = yf.Ticker("BTCUSD=X")
                        data = ticker.history(start=start_time, end=end_time, interval='5m')
                
                if not data.empty:
                    break
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                else:
                    raise e
        
        if data.empty:
            print("No data returned from yfinance. Sleeping 1s...")
            time.sleep(1)
            return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        
        # Reset index to get datetime as a column
        data = data.reset_index()

        # Handle timezone conversion properly
        if data['Datetime'].dt.tz is None:
            # If no timezone, assume UTC
            data['Datetime'] = data['Datetime'].dt.tz_localize('UTC')
        else:
            # Convert to UTC
            data['Datetime'] = data['Datetime'].dt.tz_convert('UTC')
        
        # Rename columns to match Binance format
        data = data.rename(columns={
            'Datetime': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Convert datetime to timestamp (milliseconds) and format string
        data['timestamp'] = data['datetime'].astype(np.int64) // 10**6  # Convert to milliseconds
        data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure all columns are strings for consistency with Binance format
        data['open'] = data['open'].astype(str)
        data['high'] = data['high'].astype(str)
        data['low'] = data['low'].astype(str)
        data['close'] = data['close'].astype(str)
        data['volume'] = data['volume'].astype(str)
        
        return data[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        print(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])


# === Fetch OHLCV from CoinMetrics ===
def get_coinmetrics_ohlcv(symbol="btc-usd-binance", interval="1m", start_time=None, limit=1000):
    """
    Fetch OHLCV data from CoinMetrics in a format similar to Binance/yfinance
    """
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(minutes=limit)
    end_time = start_time + timedelta(minutes=limit)


    url = "https://api.coinmetrics.io/v4/timeseries/market-candles"
    params = {
        "markets": symbol,
        "frequency": interval,
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json().get("data", [])


        if not data:
            return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])


        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["time"]).astype("int64") // 10**6
        df["datetime"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")


        # Match Binance format
        df = df.rename(columns={
            "price_open": "open",
            "price_high": "high",
            "price_low": "low",
            "price_close": "close",
            "sum_volume": "volume"
        })


        return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


    except Exception as e:
        print(f"Error fetching data from CoinMetrics: {e}")
        return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])

# === Get Last Timestamp from File if Exists ===
def get_last_saved_timestamp(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, usecols=["timestamp"])
        if not df.empty:
            last_ts = int(df["timestamp"].iloc[-1])
            return datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(
                minutes=5
            )
    except Exception:
        pass
    return None


def start_download():
    end_date = datetime.now(timezone.utc)
    
    # go one folder up and find predict_dataset.csv
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    save_file = os.path.join(parent_dir, "data", "Bitcoin_predict_dataset.csv")
    
    current = get_last_saved_timestamp(save_file)
    print("last timestamp", current)
    print("end_date", end_date)

    while current < end_date:
        try:
            print(f"Fetching from {current.isoformat()} ...")
            # df = get_yfinance_ohlcv(symbol, interval, current)
            df = get_binance_ohlcv(symbol, interval, current)
            if df.empty:
                print("No data returned. Sleeping 1s...")
                time.sleep(1)
                continue

            # Avoid duplicates
            if os.path.exists(save_file):
                existing = pd.read_csv(save_file)
                df = df[~df["timestamp"].isin(existing["timestamp"])]
            else:
                existing = pd.DataFrame()

            if df.empty:
                # No new rows
                current += timedelta(minutes=1)
                continue

            # Merge with existing to calculate rolling window correctly
            combined = pd.concat([existing, df], ignore_index=True)

            # Ensure numeric columns
            combined["low"] = pd.to_numeric(combined["low"])
            combined["high"] = pd.to_numeric(combined["high"])

            # Compute rolling 1h low/high
            combined["low_1h"] = combined["low"].rolling(window=60).min()
            combined["high_1h"] = combined["high"].rolling(window=60).max()

            # Keep only the new rows to append
            new_rows = combined.iloc[len(existing):]

            # Append to CSV
            new_rows.to_csv(
                save_file, mode="a", header=not os.path.exists(save_file), index=False
            )

            # Move forward
            last_ts = int(df["timestamp"].iloc[-1])
            current = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(minutes=1)
            time.sleep(0.2)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

    print("Data download completed.")


# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'low_1h', 'high_1h']
FORECAST_LENGTH = 12 # 5 * 12 = 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 60 * 5 = 300 mins sequence length


def invert_normalization_multi(scaler, data, n_features):
    """
    Inverse transform for multi-output predictions.
    data: (FORECAST_LENGTH, 3) → close, low, high
    """
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, 3] = data[:, 0]  # close
    dummy[:, 5] = data[:, 1]  # low
    dummy[:, 6] = data[:, 2]  # high
    return scaler.inverse_transform(dummy)[:, [3, 5, 6]]


def predict(filename):
    """
    Load trained model and predict next 60 mins (close, low, high).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    # csv_path = os.path.join(parent_dir, "data", "Bitcoin.csv")
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_predict_dataset.csv")
    model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_multi.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Provide dataset first.")

    # Load model & data
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    # Take last SEQ_LENGTH timesteps
    last_sequence = data_normalized[-SEQ_LENGTH:]
    X_input = np.expand_dims(last_sequence, axis=0)  # shape (1, SEQ_LENGTH, n_features)

    # Predict
    prediction = model.predict(X_input)  # shape (1, FORECAST_LENGTH*3)
    prediction = prediction.reshape((FORECAST_LENGTH, 3))  # (12, 3)

    # Inverse transform
    prediction_original = invert_normalization_multi(scaler, prediction, len(FEATURES))

    # Convert to DataFrame
    df_pred = pd.DataFrame(prediction_original, columns=["close", "low_1h", "high_1h"])

    # Get current time % 5
    now = datetime.now(timezone.utc)
    # Round minutes down to nearest multiple of 5
    rounded_minutes = (now.minute // 5) * 5

    # Create a new datetime object with rounded values
    now = now.replace(
        minute=rounded_minutes,
        second=0,
        microsecond=0
    )
    df_pred["timestamp"] = [now + timedelta(minutes=i * 5) for i in range(1, FORECAST_LENGTH + 1)]

    print("df pred", df_pred)
    # Reorder columns for clarity
    df_pred = df_pred[["timestamp", "close", "low_1h", "high_1h"]]
    
    # === Save only the 1-hour ahead prediction ===
    last_pred = df_pred.tail(1).copy()
    
    print("df tail", last_pred)
    
    # Final point value & interval value
    mae_low = 152
    mae_high = 159
    alpha_low = 0.12
    alpha_high = 0.12
    
    final_point = last_pred["close"]
    final_interval_high = last_pred["high_1h"] + mae_high * alpha_high
    final_interval_low = last_pred["low_1h"] - mae_low * alpha_low

    # Assume last_pred is your dataframe with 1 row (the last forecast)
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)

        # Check if timestamp already exists
        if str(last_pred["timestamp"].iloc[0]) in df_existing["timestamp"].astype(str).values:
            print(f"{last_pred['timestamp']} data already exists, skipping append.")
        else:
            last_pred.to_csv(filename, mode="a", header=False, index=False)
            print(f"{last_pred['timestamp']} data row appended.")
    else:
        # Create new file with header
        last_pred.to_csv(filename, index=False)

    return True


def wait_until_next_run():
    """
    Wait until the next time of format: every 5 minutes + 1 second
    Example: 00:04:01, 00:09:01, 00:14:01 ...
    """
    now = datetime.utcnow()
    # Next minute aligned to (current minute // 5 + 1) * 5 + 4
    # next_minute = now.minute + 1
    next_minute = (now.minute // 5 + 1) * 5
    next_time = now.replace(second=30, microsecond=0)

    if next_minute >= 60:
        next_time = next_time.replace(minute=next_minute - 60) + timedelta(hours=1)
    else:
        next_time = next_time.replace(minute=next_minute)

    # Sleep until next run
    wait_seconds = (next_time - datetime.utcnow()).total_seconds()
    if wait_seconds > 0:
        print(f"[Scheduler] Waiting {int(wait_seconds)}s until {next_time.strftime('%H:%M:%S')}")
        time.sleep(wait_seconds)


def run_scheduler():
    while True:
        wait_until_next_run()
        print("Start prediction at", datetime.utcnow().strftime("%H:%M:%S"))
        start_download()
        time.sleep(1)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        pred_csv_path = os.path.join(parent_dir, "data", "btc_1hour_ahead_prediction.csv")
        
        predict(pred_csv_path)
        print("End prediction at", datetime.utcnow().strftime("%H:%M:%S"))

if __name__ == "__main__":
    print("[Scheduler] Starting... will run every 5 minutes at mm:04:01, mm:09:01, mm:14:01 ...")
    run_scheduler()