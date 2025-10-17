import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

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
    model_path = os.path.join(parent_dir, "models", "BTC_5min_cpu_nodrop_0911.h5")
    csv_path = os.path.join(parent_dir, "data", "ff.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Convert timestamp to datetime if needed
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    
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
    split = -688
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

    def calc_interval_score(mae_low, mae_high):
        interval_scores = np.zeros(length)
        fi_score = np.zeros(length)
        fw_score = np.zeros(length)
        for i in range(0, length):
            custom_high = y_pred_high[i] + mae_high
            custom_low = y_pred_low[i] - mae_low

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
            
        sum_50 = sum(1 for v in interval_scores if v > 0.5)
        sum_75 = sum(1 for v in interval_scores if v > 0.75)
        sum_90 = sum(1 for v in interval_scores if v > 0.9)
        
        return sum_50, sum_75, sum_90

    score_panel = []
    for alpha_low in range(-200, 100):
        for alpha_high in range(-200, 100):
            score_50, score_75, score_90 = calc_interval_score(alpha_low, alpha_high)
            score_panel.append({"low": alpha_low, "high": alpha_high, "score_50": score_50, "score_75": score_75, "score_90": score_90})
    
    # Convert to DataFrame
    df_scores = pd.DataFrame(score_panel)
    # Save to CSV
    df_scores.to_csv("score_panel_10.csv", index=False)
    

if __name__ == "__main__":
    print(f"Evaluating multi-output GRU model for BTC-USD (close, low, high)...")
    evaluate_model(plot=True)
    