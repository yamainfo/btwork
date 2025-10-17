import pandas as pd
import ta
import time
from datetime import datetime, timedelta, timezone
import os

load_file_name = "datasets/1_btc_1min_1year.csv"
save_file_name="datasets/1_btc_1min_1year_trend.csv"

###Load your CSV file, add trend column
df = pd.read_csv(load_file_name)

# Check that the 'close' column exists
if 'close' not in df.columns:
    raise ValueError("The CSV file must contain a 'close' column.")

# Create the 'trend' column
df['trend'] = -1
df.loc[60:, 'trend'] = (df['close'][60:] > df['close'].shift(60)[60:]).astype(int)

### Start ADD EMA & RSI

df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
df['ema_30'] = ta.trend.ema_indicator(df['close'], window=30)
df['ema_60'] = ta.trend.ema_indicator(df['close'], window=60)
df['ema_120'] = ta.trend.ema_indicator(df['close'], window=120)

df['ema_diff_60_10'] = df['ema_60'] - df['ema_10']
df['ema_diff_120_60'] = df['ema_120'] - df['ema_60']

df['rsi_5'] = ta.momentum.rsi(df['close'], window=5)
df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
df['rsi_30'] = ta.momentum.rsi(df['close'], window=30)
df['rsi_60'] = ta.momentum.rsi(df['close'], window=60)

### End ADD EMA & RSI

# Save the modified CSV file
df.to_csv(save_file_name, index=False)

print(f"Saved as ", save_file_name)


