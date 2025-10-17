import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from datetime import timedelta

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'low_1h', 'high_1h']
FORECAST_LENGTH = 12 # 5 * 12 = 60 mins ahead forecast
SEQ_LENGTH = 60 # Last 60 * 5 = 300 mins sequence length


def create_sequences_multi(data):
    """
    Create sequences for multiple target prediction: close, low_1h, high_1h
    """
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - FORECAST_LENGTH):
        X.append(data[i:(i + SEQ_LENGTH)])
        # Select columns: close=3, low=2, high=1
        y.append(data[(i + SEQ_LENGTH):(i + SEQ_LENGTH + FORECAST_LENGTH), [3, 5, 6]])
    return np.array(X), np.array(y)  # y shape: (samples, FORECAST_LENGTH, 3)


def invert_normalization_multi(scaler, data, n_features):
    """
    Inverse transform for multi-output predictions
    data: (samples, FORECAST_LENGTH, 3)
    """
    inv_data = []
    for i in range(data.shape[0]):
        dummy = np.zeros((data.shape[1], n_features))
        dummy[:, 3] = data[i, :, 0]  # close
        dummy[:, 5] = data[i, :, 1]  # low_1h
        dummy[:, 6] = data[i, :, 2]  # high_1h
        inv_data.append(scaler.inverse_transform(dummy)[:, [3, 5, 6]])
    return np.array(inv_data)


def evaluate_model(plot: bool = True):
    """
    Load trained model and evaluate on last 20% of dataset.
    Returns metrics dict.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # build path to data folder
    model_path = os.path.join(parent_dir, "models", "BTC_5min_0918.h5")
    csv_path = os.path.join(parent_dir, "data", "dataset_5min_cm.csv")
    sec_path = os.path.join(parent_dir, "data", "dataset_second_cm.csv")
    sec_df = pd.read_csv(sec_path)
    sec_df["datetime"] = pd.to_datetime(sec_df["datetime"], utc=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Convert timestamp to datetime if needed
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    
    mins = data[FEATURES].min()
    maxs = data[FEATURES].max()

    print("Minimum values:\n", mins)
    print("Maximum values:\n", maxs)
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences_multi(data_normalized)

    # Flatten y for evaluation
    y_flat = y.reshape((y.shape[0], FORECAST_LENGTH * 3))

    # Test split (last 1 day)
    # split = int(0.99 * len(X))
    split = -288
    X_val, y_val = X[split:], y[split:]

    n_features = len(FEATURES)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    # Predict
    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    predictions_val_original = invert_normalization_multi(scaler, predictions_val, n_features)
    y_val_original = invert_normalization_multi(scaler, y_val, n_features)
    
    # Create timestamps aligned with each prediction sample
    total_samples = len(data_normalized)
    aligned_timestamps = data["timestamp"].iloc[SEQ_LENGTH + FORECAST_LENGTH - 1 : SEQ_LENGTH + FORECAST_LENGTH - 1 + total_samples]
    aligned_timestamps = aligned_timestamps.reset_index(drop=True)
    
    ts = aligned_timestamps[split: -1]

    # Evaluate interval score for each 5-minute interval
    length = len(aligned_timestamps[split: -1])
    y_true_low = y_val_original[:, -1, 1]   # last step
    y_pred_low = predictions_val_original[:, -1, 1]
    y_true_high = y_val_original[:, -1, 2]   # last step
    y_pred_high = predictions_val_original[:, -1, 2]
    y_true_close = y_val_original[:, -1, 0]   # last step

    interval_scores = np.zeros(length)
    fi_score = np.zeros(length)
    fw_score = np.zeros(length)
    
    # custom value to manipulate.
    mae_low = 30
    mae_high = 20
    alpha_low = -1
    alpha_high = -1
    for i in range(0, length):
        custom_high = y_pred_high[i] + mae_high * alpha_high
        custom_low = y_pred_low[i] - mae_low * alpha_low

        effective_top = min(y_true_high[i], custom_high)
        effective_bottom = max(y_true_low[i], custom_low)
        fw = (effective_top - effective_bottom) / (custom_high - custom_low)
        if (fw < 0 or fw > 1):
            fw = 0
        fw_score[i] = fw

        fi = 0
        if (i >= 12):
            hour_slice = sec_df[
                (sec_df["datetime"] >= ts.iloc[i]) &
                (sec_df["datetime"] <= ts.iloc[i] + timedelta(hours=1))
            ]
            hour_prices = hour_slice["ReferenceRateUSD"].tolist()
            prices_in_bounds = sum(1 for price in hour_prices if custom_low <= price <= custom_high)
            fi = prices_in_bounds / len(hour_prices)
            fi_score[i] = fi
        
        interval_scores[i] = fi * fw

    print(len(interval_scores))
    print("average interval scores:", sum(v for v in interval_scores) / len(interval_scores))
    print("Total interval count > 0.9:", sum(1 for v in interval_scores if v >= 0.9))
    print("Total interval scores > 0.9:", sum(v for v in interval_scores if v >= 0.9))
    print("Total interval count 0.9 > v >= 0.75:", sum(1 for v in interval_scores if 0.9 > v >= 0.75))
    print("Total interval scores 0.9 > v >= 0.75:", sum(v for v in interval_scores if 0.9 > v >= 0.75))
    print("Total interval count 0.75 > v >= 0.5:", sum(1 for v in interval_scores if 0.75 > v >= 0.5))
    print("Total interval scores 0.75 > v >= 0.5:", sum(v for v in interval_scores if 0.75 > v >= 0.5))
    print("Total interval count 0.5 > v >= 0.25:", sum(1 for v in interval_scores if 0.5 > v >= 0.25))
    print("Total interval scores 0.5 > v >= 0.25:", sum(v for v in interval_scores if 0.5 > v >= 0.25))
    print("Total interval count 0.25 > v >= 0.1:", sum(1 for v in interval_scores if 0.25 > v >= 0.1))
    print("Total interval scores 0.25 > v >= 0.1:", sum(v for v in interval_scores if 0.25 > v >= 0.1))
    print("Total interval count < 0.1:", sum(1 for v in interval_scores if v < 0.1))
    print("Total interval scores < 0.1:", sum(v for v in interval_scores if v < 0.1))
    
    if plot:
        # --- Create a figure with 4 rows: fi/fw/interval + 3 metrics ---
        fig, axes = plt.subplots(4, 1, figsize=(120, 160), sharex=True)
        fig.subplots_adjust(hspace=0.4)  # spacing between plots

        # 1️⃣ Plot fi, fw, interval scores
        axes[0].plot(ts, fi_score, label="FI Scores")
        axes[0].plot(ts, fw_score, label="FW Scores")
        axes[0].plot(ts, interval_scores, label="Interval Scores")
        axes[0].set_ylabel("Score (0-1)")
        axes[0].legend()
        axes[0].set_title("FI / FW / Interval Scores")

        # 2️⃣ Plot each target metric
        target_names = ["close", "low_1h", "high_1h"]

        for idx, name in enumerate(target_names):
            ts = aligned_timestamps[split:]
            y_true = y_val_original[:, -1, idx]
            y_pred = predictions_val_original[:, -1, idx]

            axes[idx + 1].plot(ts, y_true, label=f"Actual {name}")
            axes[idx + 1].plot(ts, y_pred, label=f"Predicted {name}")
            axes[idx + 1].set_ylabel("Price (USD)")
            axes[idx + 1].legend()
            axes[idx + 1].set_title(f"BTC-USD 60min Forecast Evaluation ({name})")

        # Final x-axis label
        axes[-1].set_xlabel("Time Steps")

        # Show all plots together
        plt.show()
    
    variation = np.zeros(len(y_val_original[:, -1, 0]))
    for i in range(0, len(y_val_original[:, -1, 0])):
        variation[i] = abs(y_val_original[:, -1, 0][i] - predictions_val_original[:, -1, 0][i])
    
    print("mae_value", np.sum(variation) / len(variation))
    print("mae_absoute_error", mean_absolute_error(y_val_original[:, -1, 0], predictions_val_original[:, -1, 0]))
    # Save to CSV
    ts = aligned_timestamps[split:]
    df = pd.DataFrame({
        "datetime": ts,
        "variation": variation
    })
    df.to_csv("variation.csv", index=False)

    metrics = {}
    for idx, name in enumerate(target_names):
        y_true = y_val_original[:, -1, idx]   # last step
        y_pred = predictions_val_original[:, -1, idx]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        normalized_rmse = rmse / (y_true.max() - y_true.min())

        metrics[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "normalized_rmse": normalized_rmse,
        }
        
    
    return metrics



if __name__ == "__main__":
    print(f"Evaluating multi-output GRU model for BTC-USD (close, low, high)...")
    results = evaluate_model(plot=True)
    print("Metrics:", results)