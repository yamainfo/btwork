import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os

# go one folder up
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# build path to data folder
load_file_path = os.path.join(parent_dir, "data", "Bitcoin.csv")
save_file_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")

###Load your CSV file, add trend column
df = pd.read_csv(load_file_path)

# Check that the 'close' column exists
if 'close' not in df.columns:
    raise ValueError("The CSV file must contain a 'close' column.")

# low_1h and high_1h
df['low_1h'] = df['low'].rolling(window=12).min()
df['high_1h'] = df['high'].rolling(window=12).max()

# Save the modified CSV file
df.to_csv(save_file_path, mode="a", header=not os.path.exists(save_file_path), index=False)

print(f"Saved as ", save_file_path)


