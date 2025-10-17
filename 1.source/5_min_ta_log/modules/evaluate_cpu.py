import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from datetime import timedelta

# -------------------- Config --------------------
FEATURES = [
    'volume',      # log-transformed volume
    'log_close',
    'log_low_1h',
    'log_high_1h',
    'ema5_log',
    'ema12_log',
    'ema24_log',
    'ema48_log',
    'ema_5_min_24',
    'ema_12_min_48',
    'close_vol12',
    'close_vol60',
    'slope12',
    'slope36',
    'stoch_rsi',
    'macd_signal',
    'macd_hist'
]
FORECAST_LENGTH = 12  # 60 mins ahead
SEQ_LENGTH = 60       # Last 300 mins

# -------------------- Helpers --------------------
def create_sequences_multi(data):
    """
    Create sequences for multiple target prediction: dlog_close, dlog_low_1h, dlog_high_1h
    """
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - FORECAST_LENGTH):
        X.append(data[i:(i + SEQ_LENGTH)])
        y.append(data[(i + SEQ_LENGTH):(i + SEQ_LENGTH + FORECAST_LENGTH), [1, 2, 3]])  # dlog_* targets
    return np.array(X), np.array(y)


def invert_normalization_multi(scaler, data, n_features):
    """
    Inverse transform for multi-output predictions (dlog_close, dlog_low_1h, dlog_high_1h)
    Returns values in original scale (dlog units).
    """
    inv_data = []
    for i in range(data.shape[0]):
        dummy = np.zeros((data.shape[1], n_features))
        dummy[:, 1] = data[i, :, 0]  # close
        dummy[:, 2] = data[i, :, 1]  # low_1h
        dummy[:, 3] = data[i, :, 2]  # high_1h
        inv_data.append(scaler.inverse_transform(dummy)[:, [1, 2, 3]])
    return np.array(inv_data)

# -------------------- Evaluation --------------------
def evaluate_model(plot=True):
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    model_path = os.path.join(parent_dir, "models", "BTC_5min_ta_complex.h5")
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")
    sec_path = os.path.join(parent_dir, "data", "dataset_second_cm.csv")

    if not os.path.exists(model_path) or not os.path.exists(csv_path) or not os.path.exists(sec_path):
        raise FileNotFoundError("Model or CSV data missing.")

    model = load_model(model_path)
    data = pd.read_csv(csv_path)
    sec_df = pd.read_csv(sec_path)
    sec_df["datetime"] = pd.to_datetime(sec_df["datetime"], utc=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)

    scaler = MinMaxScaler(feature_range=(0,1))
    data_normalized = scaler.fit_transform(data[FEATURES])
    X, y = create_sequences_multi(data_normalized)

    split = -288
    X_val, y_val = X[split:], y[split:]
    n_features = len(FEATURES)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    pred_log = invert_normalization_multi(scaler, predictions_val, n_features)
    true_log = invert_normalization_multi(scaler, y_val, n_features)


    # Create timestamps aligned with each prediction sample
    total_samples = len(data_normalized)
    aligned_timestamps = data["timestamp"].iloc[SEQ_LENGTH + FORECAST_LENGTH - 1 : SEQ_LENGTH + FORECAST_LENGTH - 1 + total_samples]
    aligned_timestamps = aligned_timestamps.reset_index(drop=True)
    
    ts = aligned_timestamps[split: -1]

    # Evaluate interval score for each 5-minute interval
    length = len(aligned_timestamps[split: -1])
    y_true_low = np.exp(true_log[:, -1, 1])
    y_pred_low = np.exp(pred_log[:, -1, 1])
    
    y_true_high = np.exp(true_log[:, -1, 2])
    y_pred_high = np.exp(pred_log[:, -1, 2])
    
    y_true_close = np.exp(true_log[:, -1, 0])

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

    print(len(interval_scores))
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
        axes[0].set_title("FI / FW / Interval Scores with Log")

        # 2️⃣ Plot each target metric
        target_names = ["close", "low_1h", "high_1h"]

        for idx, name in enumerate(target_names):
            ts = aligned_timestamps[split:]
            y_true = np.exp(true_log[:, -1, idx])
            y_pred = np.exp(pred_log[:, -1, idx])

            axes[idx + 1].plot(ts, y_true, label=f"Actual {name}")
            axes[idx + 1].plot(ts, y_pred, label=f"Predicted {name}")
            axes[idx + 1].set_ylabel("Price (USD)")
            axes[idx + 1].legend()
            axes[idx + 1].set_title(f"BTC-USD 60min Forecast Evaluation ({name}) with Log")

        # Final x-axis label
        axes[-1].set_xlabel("Time Steps")

        # Show all plots together
        plt.show()
    
    variation = np.zeros(len(true_log[:, -1, 0]))
    for i in range(0, len(true_log[:, -1, 0])):
        variation[i] = abs(np.exp(true_log[:, -1, 0][i]) - np.exp(pred_log[:, -1, 0][i]))
    
    print("mae_value", np.sum(variation) / len(variation))
    print("mae_absoute_error", mean_absolute_error(true_log[:, -1, 0], pred_log[:, -1, 0]))
    # Save to CSV
    ts = aligned_timestamps[split:]
    df = pd.DataFrame({
        "datetime": ts,
        "variation": variation
    })
    df.to_csv("variation.csv", index=False)

    metrics = {}
    for idx, name in enumerate(target_names):
        y_true = np.exp(true_log[:, -1, idx])   # last step
        y_pred = np.exp(pred_log[:, -1, idx])

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