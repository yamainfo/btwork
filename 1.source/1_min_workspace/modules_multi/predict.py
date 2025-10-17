import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'low_1h', 'high_1h']
FORECAST_LENGTH = 60  # 60 mins ahead forecast
SEQ_LENGTH = 120      # Last 120 mins sequence length


def invert_normalization_multi(scaler, data, n_features):
    """
    Inverse transform for multi-output predictions.
    data: (FORECAST_LENGTH, 3) → close, low, high
    """
    dummy = np.zeros((data.shape[0], n_features))
    print("-----------------------", data)
    dummy[:, 3] = data[:, 0]  # close
    dummy[:, 5] = data[:, 1]  # low
    dummy[:, 6] = data[:, 2]  # high
    return scaler.inverse_transform(dummy)[:, [3, 5, 6]]


def predict_next():
    """
    Load trained model and predict next 60 mins (close, low, high).
    """
    model_path = "models/GRU_BTC_60_multi.h5"
    csv_path = "data/Bitcoin_training_dataset.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Provide dataset first.")

    # Load model & data
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    # Take last SEQ_LENGTH timesteps
    last_sequence = data_normalized[-SEQ_LENGTH:]
    X_input = np.expand_dims(last_sequence, axis=0)  # shape (1, SEQ_LENGTH, n_features)

    # Predict
    prediction = model.predict(X_input)  # shape (1, FORECAST_LENGTH*3)
    prediction = prediction.reshape((FORECAST_LENGTH, 3))  # (60, 3)
    print(prediction)

    # Inverse transform
    prediction_original = invert_normalization_multi(scaler, prediction, len(FEATURES))

    # Convert to DataFrame
    df_pred = pd.DataFrame(prediction_original, columns=["close", "low_1h", "high_1h"])
    df_pred.index = pd.RangeIndex(start=1, stop=FORECAST_LENGTH+1, step=1)  # minutes ahead

    print("\n=== Next 60 Minutes Forecast (BTC-USD) ===")
    print(df_pred.head(10))  # show first 10 mins
    print("...")
    print(df_pred.tail(5))   # show last 5 mins

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df_pred["close"], label="Predicted Close")
    plt.plot(df_pred["low_1h"], label="Predicted Low")
    plt.plot(df_pred["high_1h"], label="Predicted High")
    plt.title("BTC-USD Next 60 Minutes Forecast")
    plt.xlabel("Minutes Ahead")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    return df_pred


if __name__ == "__main__":
    forecast = predict_next()
    forecast.to_csv("data/forecast_next_60min.csv", index_label="minute")
    print("\nSaved forecast to data/forecast_next_60min.csv")
