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
    # go one folder up
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # build path to data folder
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")
    model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_multi.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load
    # Load
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences_multi(data_normalized)

    # Flatten y for evaluation
    y_flat = y.reshape((y.shape[0], FORECAST_LENGTH * 3))

    # Test split (last 3%)
    split = int(0.97 * len(X))
    X_val, y_val = X[split:], y[split:]

    n_features = len(FEATURES)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    # Predict
    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    predictions_val_original = invert_normalization_multi(scaler, predictions_val, n_features)
    y_val_original = invert_normalization_multi(scaler, y_val, n_features)

    # Evaluate separately for each target (last timestep of 60-min horizon)
    metrics = {}
    target_names = ["close", "low_1h", "high_1h"]

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

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label=f"Actual {name}")
            plt.plot(y_pred, label=f"Predicted {name}")
            plt.title(f"BTC-USD 60min Forecast Evaluation ({name})")
            plt.xlabel("Time steps")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.show()

    return metrics


if __name__ == "__main__":
    print(f"Evaluating multi-output GRU model for BTC-USD (close, low, high)...")
    results = evaluate_model(plot=True)
    print("Metrics:", results)