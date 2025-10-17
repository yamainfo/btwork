import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import os
import ta
from sklearn.linear_model import LinearRegression
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochRSIIndicator

# go one folder up
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# build path to data folder
load_file_path = os.path.join(parent_dir, "data", "dataset_5min_cm.csv")
save_file_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")

###Load your CSV file, add trend column
df = pd.read_csv(load_file_path)

# Check that the 'close' column exists
if 'close' not in df.columns:
    raise ValueError("The CSV file must contain a 'close' column.")

# low_1h and high_1h
df['low_1h'] = df['low'].rolling(window=12).min()
df['high_1h'] = df['high'].rolling(window=12).max()

# add log_return close, high, low price
df["log_high"] = np.log(df["high"])
df["log_low"] = np.log(df["low"])
df["log_close"] = np.log(df["close"])
df["log_low_1h"] = np.log(df["low_1h"])
df["log_high_1h"] = np.log(df["high_1h"])

# MA
# df[f'ma5_log'] = df['log_close'].rolling(5, min_periods=1).mean().shift(1)
# df[f'ma12_log'] = df['log_close'].rolling(12, min_periods=1).mean().shift(1)
# df[f'ma36_log'] = df['log_close'].rolling(36, min_periods=1).mean().shift(1)

# EMA
df[f'ema5_log'] = df['log_close'].ewm(span=5, adjust=False).mean().shift(1)
df[f'ema12_log'] = df['log_close'].ewm(span=12, adjust=False).mean().shift(1)
df[f'ema24_log'] = df['log_close'].ewm(span=24, adjust=False).mean().shift(1)
df[f'ema48_log'] = df['log_close'].ewm(span=36, adjust=False).mean().shift(1)

# EMA Diff
df["ema_5_min_24"] = df["ema5_log"] - df["ema24_log"]
df["ema_12_min_48"] = df["ema12_log"] - df["ema48_log"]

# Rolling Volatility
df['close_vol12'] = df['log_close'].rolling(12).std().shift(1)
df['close_vol60'] = df['log_close'].rolling(60).std().shift(1)
# df['low_vol12'] = df['log_low_1h'].rolling(12).std().shift(1)
# df['low_vol36'] = df['log_low_1h'].rolling(36).std().shift(1)
# df['low_vol60'] = df['log_low_1h'].rolling(60).std().shift(1)
# df['high_vol12'] = df['log_high_1h'].rolling(12).std().shift(1)
# df['high_vol36'] = df['log_high_1h'].rolling(36).std().shift(1)
# df['high_vol60'] = df['log_high_1h'].rolling(60).std().shift(1)

# Slope
def slope(series):
    y = series.values
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

df['slope12'] = df['log_close'].rolling(12).apply(slope, raw=False).shift(1)
df['slope36'] = df['log_close'].rolling(36).apply(slope, raw=False).shift(1)

# RSI (shifted)
# df['rsi14'] = RSIIndicator(df['close'], window=14).rsi().shift(1)

# StochRSI (defaults: window=14, smooth1=3, smooth2=3)
df['stoch_rsi'] = StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3).stochrsi().shift(1)

# MACD (macd line, signal line, histogram)
macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
df['macd'] = macd.macd().shift(1)
df['macd_signal'] = macd.macd_signal().shift(1)
df['macd_hist'] = macd.macd_diff().shift(1)  # macd - signal

# ATR and normalized ATR
atr = AverageTrueRange(df['log_high_1h'], df['log_low_1h'], df['log_close'], window=12)
df['atr12'] = atr.average_true_range().shift(1)
df['atr12_norm'] = df['atr12'] / df['log_close'].shift(1)

df.fillna(0, inplace=True)

# Save the modified CSV file
df.to_csv(save_file_path, mode="a", header=not os.path.exists(save_file_path), index=False)

print(f"Saved as ", save_file_path)


