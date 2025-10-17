import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume']

FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 120 mins sequence length


def invert_normalization(scaler, data, n_features):
    dummy = np.zeros((data.shape[0], n_features))
    dummy[:, 4] = data
    return scaler.inverse_transform(dummy)[:, 4]


def predict():
    """
    Load a saved GRU model and forecast the next N minutes.
    """
    model_path = f"models/GRU_Bitcion_60.h5"
    csv_path = f"data/Bitcion.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Download data first.")

    # Load model and data
    model = load_model(model_path)
    data = pd.read_csv(csv_path)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    last_sequence = data_normalized[-SEQ_LENGTH:]
    last_sequence = last_sequence.reshape(1, SEQ_LENGTH, len(FEATURES))

    # Predict
    predictions_future = model.predict(last_sequence)
    predictions_future_original = invert_normalization(
        scaler, predictions_future.flatten(), len(FEATURES)
    )

    return predictions_future_original


if __name__ == "__main__":
    print(f"Predicting next BTC 1 hour ahead price...")
    
    forecast = predict()
    
    print(forecast)
