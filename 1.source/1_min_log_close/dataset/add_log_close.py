import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import os
import ta

# go one folder up
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# build path to data folder
load_file_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")
save_file_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset_log_close.csv")

###Load your CSV file, add trend column
df = pd.read_csv(load_file_path)

# Check that the 'close' column exists
if 'close' not in df.columns:
    raise ValueError("The CSV file must contain a 'close' column.")

# add log_return close price
df["log_close"] = np.log(df["close"] / df["close"].shift(1))

# df["return_5m"] = df["close"].pct_change()
# df["volatility"] = df["return_5m"].rolling(12).std()

# df["ema_fast"] = df["close"].ewm(span=12).mean()
# df["ema_slow"] = df["close"].ewm(span=48).mean()
# df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

# df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
# df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"], window=14).stochrsi()
# df["macd"] = ta.trend.MACD(df["close"]).macd()
# df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

df.fillna(0, inplace=True)

# Save the modified CSV file
df.to_csv(save_file_path, mode="a", header=not os.path.exists(save_file_path), index=False)

print(f"Saved as ", save_file_path)


