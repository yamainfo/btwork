import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume']
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 60 # Last 120 mins sequence length


def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - FORECAST_LENGTH):
        X.append(data[i:(i + SEQ_LENGTH)])
        y.append(data[(i + SEQ_LENGTH):(i + SEQ_LENGTH + FORECAST_LENGTH), 3])  # Close price
    return np.array(X), np.array(y)


def invert_normalization(scaler, data, n_features):
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, 3] = data
    return scaler.inverse_transform(dummy)[:, 3]


def evaluate_model(plot: bool = True):
    """
    Load trained model and evaluate on last 20% of dataset.
    Returns metrics dict.
    """
    # go one folder up
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # build path to data folder
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")
    model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_single.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load
    model = load_model(model_path)
    data = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime if needed
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences(data_normalized)
    
    split = int(0.97 * len(X))
    X_val, y_val = X[split:], y[split:]

    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(FEATURES)))

    # Predict
    predictions_val = model.predict(X_val)
    predictions_val_original = invert_normalization(scaler, predictions_val[:, -1], len(FEATURES))
    y_val_original = invert_normalization(scaler, y_val[:, -1], len(FEATURES))
    
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
    
    # Filter only 5-minute indices
    y_true = y_val_original[valid_idx]
    y_pred = predictions_val_original[valid_idx]
    ts = val_timestamps.iloc[valid_idx]

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    normalized_rmse = rmse / (y_true.max() - y_pred.min())

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "normalized_rmse": normalized_rmse,
    }

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(ts, y_true, label=f"Actual Price")
        plt.plot(ts, y_pred, label=f"Predicted Pric")
        plt.title(f"BTC-USD 60min Forecast Evaluation Single Training")
        plt.xlabel("Timestamp")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

    return metrics


if __name__ == "__main__":
    forecast_length = 60
    print(f"Evaluating GRU model for BTC-USD...")
    
    results = evaluate_model(plot=True)
    print("Metrics:", results)
