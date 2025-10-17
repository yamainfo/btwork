import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'volume', 'low_1h', 'high_1h', 'log_close']
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 120 mins sequence length
MODEL_NAME = "GRU_BTC_60_1min.h5"


# -------------------- Helpers --------------------
def create_sequences_multi(data):
    """
    Create sequences for multiple target prediction: log_close, low_1h, high_1h
    """
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - FORECAST_LENGTH):
        X.append(data[i:(i + SEQ_LENGTH)])
        # Select columns: log_close=6, low_1h=4, high_1h=5
        y.append(data[(i + SEQ_LENGTH):(i + SEQ_LENGTH + FORECAST_LENGTH), [6, 4, 5]])
    return np.array(X), np.array(y)  # y shape: (samples, FORECAST_LENGTH, 3)


def invert_normalization_multi(scaler, data, n_features):
    """
    Inverse transform for multi-output predictions
    data: (samples, FORECAST_LENGTH, 3)
    """
    inv_data = []
    for i in range(data.shape[0]):
        dummy = np.zeros((data.shape[1], n_features))
        dummy[:, 6] = data[i, :, 0]  # log_close
        dummy[:, 4] = data[i, :, 1]  # low_1h
        dummy[:, 5] = data[i, :, 2]  # high_1h
        inv_data.append(scaler.inverse_transform(dummy)[:, [6, 4, 5]])
    return np.array(inv_data)  # Add missing return statement


def evaluate_model(plot: bool = True):
    """
    Load trained model and evaluate on last 20% of dataset.
    Evaluates only at 5-minute intervals.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # build path to data folder
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset_log_close.csv")
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
    split = -400
    X_val, y_val = X[split:], y[split:]
    
    print("len data", len(X), split, len(X) - split)

    n_features = len(FEATURES)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    # Predict
    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    predictions_val_original = invert_normalization_multi(scaler, predictions_val, n_features)
    y_val_original = invert_normalization_multi(scaler, y_val, n_features)

    # Reconstruct actual close prices from log returns
    pred_log_returns = predictions_val_original[:, :, 0]  # log_close predictions
    
    close_true = []
    close_pred = []
    
    # Calculate train_split equivalent for validation data
    train_split = len(data) + split - SEQ_LENGTH - FORECAST_LENGTH
    
    for i in range(len(pred_log_returns)):
        # Get starting price = last close in sequence  
        last_close = data["close"].iloc[train_split + i + SEQ_LENGTH - 1]
        
        # Convert cumulative log returns → predicted prices
        pred_close_seq = last_close * np.exp(np.cumsum(pred_log_returns[i]))
        
        # True prices from original dataset
        true_close_seq = data["close"].iloc[
            train_split + i + SEQ_LENGTH : train_split + i + SEQ_LENGTH + FORECAST_LENGTH
        ].values

        close_pred.append(pred_close_seq)
        close_true.append(true_close_seq)

    close_pred = np.array(close_pred)
    close_true = np.array(close_true)

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
    y_true_low = y_val_original[:, -1, 1][valid_idx]
    y_pred_low = predictions_val_original[:, -1, 1][valid_idx]
    y_true_high = y_val_original[:, -1, 2][valid_idx]
    y_pred_high = predictions_val_original[:, -1, 2][valid_idx]
    
    # Use reconstructed close prices for interval calculations
    y_true_close = close_true[:, -1][valid_idx]  # Use reconstructed true close

    interval_scores = np.zeros(length)
    fi_score = np.zeros(length)
    fw_score = np.zeros(length)
    
    # custom value to manipulate.
    mae_low = 121
    mae_high = 128
    alpha_low = -0.8
    alpha_high = -0.8
    for i in range(0, length):
        custom_high = y_pred_high[i] + mae_high * alpha_high
        custom_low = y_pred_low[i] - mae_low * alpha_low

        effective_top = min(y_true_high[i], custom_high)
        effective_bottom = max(y_true_low[i], custom_low)
        fi = (effective_top - effective_bottom) / (custom_high - custom_low)
        fi_score[i] = fi

        fw = 0
        if (i >= 12):
            count = sum(1 for j in range(i -12, i) if custom_low <= y_true_close[j] <= custom_high)
            fw = count / 12
            fw_score[i] = fw
        
        interval_scores[i] = fi * fw

    print("interval score count", len(interval_scores))
    # print(interval_scores)
    print("Total interval scores: ", sum(interval_scores))
    print("Total interval scores > 0.9:", sum(v for v in interval_scores if v > 0.9))
    
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
            if name == "close":
                # Use reconstructed close prices
                y_true = close_true[:, -1][valid_idx]  # Use reconstructed true close
                y_pred = close_pred[:, -1][valid_idx]  # Use reconstructed predicted close
            else:
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
        if name == "close":
            # Use reconstructed close prices for close price evaluation
            y_true = close_true[:, -1][valid_idx]
            y_pred = close_pred[:, -1][valid_idx]
        else:
            # Filter only 5-minute indices for low_1h and high_1h
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
        
        # Print close price specific metrics
        if name == "close":
            print(f"Close Price Metrics:")
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Normalized RMSE: {normalized_rmse:.4f}")

    return metrics


if __name__ == "__main__":
    print(f"Evaluating multi-output GRU model for BTC-USD (close, low, high)...")
    results = evaluate_model(plot=True)
    print("Metrics:", results)