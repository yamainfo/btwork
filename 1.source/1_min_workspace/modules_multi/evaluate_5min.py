import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'low_1h', 'high_1h']
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 120 mins sequence length
MODEL_NAME = "BTC_1min_nodrop_0912.h5"


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
    Evaluates only at 5-minute intervals.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # build path to data folder
    csv_path = os.path.join(parent_dir, "data", "a.csv")
    model_path = os.path.join(parent_dir, "models", MODEL_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Ensure timestamp column exists
    if "timestamp" not in data.columns:
        raise KeyError("Dataset must have a 'timestamp' column in UTC or local time")

    # Convert timestamp to datetime if needed
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    # Create sequences
    X, y = create_sequences_multi(data_normalized)

    # Test split (last 3%)
    # split = int(0.982 * len(X))
    split = -140
    X_val, y_val = X[split:], y[split:]
    
    print("len data", len(X), split, len(X) - split)

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

    print("SEQ_LENGTH + FORECAST_LENGTH - 1 + total_samples", SEQ_LENGTH + FORECAST_LENGTH - 1 + total_samples)
    print("split", split)
    # Now slice validation timestamps properly
    val_timestamps = aligned_timestamps[split:]

    # Select only 5-minute intervals
    five_min_idx = np.where(val_timestamps.dt.minute % 5 == 0)[0]
    valid_idx = five_min_idx[five_min_idx < y_val_original.shape[0]]
    print("five_min_idx count:", len(five_min_idx))
    
    # Evaluate interval score for each 5-minute interval
    ts = val_timestamps.iloc[valid_idx]
    length = len(ts)
    y_true_low = y_val_original[:, -1, 1][valid_idx]   # last step
    y_pred_low = predictions_val_original[:, -1, 1][valid_idx]
    y_true_high = y_val_original[:, -1, 2][valid_idx]   # last step
    y_pred_high = predictions_val_original[:, -1, 2][valid_idx]
    y_true_close = y_val_original[:, -1, 0][valid_idx]   # last step

    interval_scores = np.zeros(length)
    fi_score = np.zeros(length)
    fw_score = np.zeros(length)
    
    # custom value to manipulate.
    mae_low = 120
    mae_high = 140
    alpha_low = -0.85
    alpha_high = -0.85
    for i in range(0, length):
        custom_high = y_pred_high[i] + mae_high * alpha_high
        custom_low = y_pred_low[i] - mae_low * alpha_low

        effective_top = min(y_true_high[i], custom_high)
        effective_bottom = max(y_true_low[i], custom_low)
        fi = (effective_top - effective_bottom) / (custom_high - custom_low)
        if (fi < 0 or fi > 1):
            fi = 0
        fi_score[i] = fi

        fw = 0
        if (i >= 12):
            count = sum(1 for j in range(i -12, i) if custom_low <= y_true_close[j] <= custom_high)
            fw = count / 12
            fw_score[i] = fw
        
        interval_scores[i] = fi * fw

    print("interval score count", len(interval_scores))
    print("Total interval scores > 0.75:", sum(1 for v in interval_scores if v > 0.75))
    print("Total interval scores > 0.75:", sum(v for v in interval_scores if v > 0.75))
    
    if plot:
        # --- Create a figure with 4 rows: fi/fw/interval + 3 metrics ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.subplots_adjust(hspace=0.4)  # spacing between plots

        # 1️⃣ Plot fi, fw, interval scores
        axes[0].plot(ts, fi_score, label="FI Scores")
        axes[0].plot(ts, fw_score, label="FW Scores")
        axes[0].plot(ts, interval_scores, label="Interval Scores")
        axes[0].set_ylabel("Score (0-1)")
        axes[0].legend()
        axes[0].set_title("FI / FW / Interval Scores - 1 min data")

        # 2️⃣ Plot each target metric
        target_names = ["close", "low_1h", "high_1h"]

        for idx, name in enumerate(target_names):
            ts = val_timestamps.iloc[valid_idx]
            y_true = y_val_original[:, -1, idx][valid_idx]
            y_pred = predictions_val_original[:, -1, idx][valid_idx]

            axes[idx + 1].plot(ts, y_true, label=f"Actual {name}")
            axes[idx + 1].plot(ts, y_pred, label=f"Predicted {name}")
            axes[idx + 1].set_ylabel("Price (USD)")
            axes[idx + 1].legend()
            axes[idx + 1].set_title(f"BTC-USD 60min Forecast Evaluation ({name}) - 1 min data")

        # Final x-axis label
        axes[-1].set_xlabel("Time Steps")

        # Show all plots together
        plt.show()

    # Evaluate separately for each target (last timestep of 60-min horizon)
    metrics = {}
    target_names = ["close", "low_1h", "high_1h"]

    for idx, name in enumerate(target_names):
        # Filter only 5-minute indices
        y_true = y_val_original[:, -1, idx][valid_idx]
        y_pred = predictions_val_original[:, -1, idx][valid_idx]
        ts = val_timestamps.iloc[valid_idx]

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