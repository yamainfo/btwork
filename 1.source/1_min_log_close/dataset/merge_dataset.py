import os
import pandas as pd

ORIGIN_FILE_NAME = "Bitcoin_training_dataset.csv"
CM_FILE_NAME = "cm.csv"
SAVE_FILE_NAME = "result.csv"

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
origin_file_path = os.path.join(parent_dir, "data", ORIGIN_FILE_NAME)
cm_file_path = os.path.join(parent_dir, "data", CM_FILE_NAME)
save_file_path = os.path.join(parent_dir, "data", SAVE_FILE_NAME)

# Read CSV files
df_a = pd.read_csv(origin_file_path)
df_b = pd.read_csv(cm_file_path)

# Convert datetime columns to proper datetime objects for both files
df_a["datetime"] = pd.to_datetime(df_a["datetime"])
df_b["datetime"] = pd.to_datetime(df_b["datetime"])

# Convert b.csv datetime to timezone-naive to match a.csv
df_b["datetime"] = df_b["datetime"].dt.tz_convert(None)

# --- Merge datasets ---
df_merged = pd.merge(
    df_a,
    df_b[["datetime", "close", "hourly_high", "hourly_low"]],
    on="datetime",
    how="left"  # keep all rows from a.csv
)

# --- Update close, high_1h, low_1h only if b.csv has data ---
df_merged["close"] = df_merged["close_y"].combine_first(df_merged["close_x"]) \
    if "close_y" in df_merged else df_merged["close"]
df_merged["high_1h"] = df_merged["hourly_high"].combine_first(df_merged["high_1h"])
df_merged["low_1h"] = df_merged["hourly_low"].combine_first(df_merged["low_1h"])

# --- Drop extra columns ---
df_merged = df_merged.drop(
    columns=[c for c in ["close_x", "close_y", "hourly_high", "hourly_low"] if c in df_merged],
    errors="ignore"
)

# --- Reorder columns: close goes after "low" and before "volume" ---
columns = list(df_merged.columns)
if "close" in columns and "low" in columns and "volume" in columns:
    columns.remove("close")  # temporarily remove
    low_index = columns.index("low")
    columns.insert(low_index + 1, "close")  # insert close after low
    df_merged = df_merged[columns]

# Drop extra columns (from merge)
df_merged = df_merged.drop(columns=["close_x", "close_y", "hourly_high", "hourly_low"], errors="ignore")

# Save back to a_new.csv
df_merged.to_csv(save_file_path, index=False)

print("✅ a_updated.csv has been created successfully!")