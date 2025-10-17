import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.DynamicReduceLROnPlateau import DynamicReduceLROnPlateau

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume']
EPOCHS = 100
BATCH_SIZE = 64
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 60 # Last 120 mins sequence length
custom_switch_factor = 0.1

# -------------------- Helpers --------------------
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


def build_gru(n_features):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)))
    model.add(Dropout(0.1))
    model.add(GRU(50))
    model.add(Dropout(0.1))
    model.add(Dense(FORECAST_LENGTH))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# -------------------- Training Function --------------------
def train_model():
    """
    Train a GRU model for a given crypto and forecast horizon.
    Saves model in models/GRU_BTC_60.h5
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_path = os.path.join(parent_dir, "data", "Bitcoin.csv")

    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download data first.")

    # Load and preprocess data
    data = pd.read_csv(csv_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences(data_normalized)

    # Train / validation split
    n = len(X)
    train_split = int(0.85 * n)
    val_split = int(0.95 * n)
    
    X_train, X_val = X[:train_split], X[train_split:val_split]
    y_train, y_val = y[:train_split], y[train_split:val_split]

    # Reshape
    n_features = len(FEATURES)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    # Build and train GRU
    model = build_gru(n_features)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Learning rate manipulation
    reduce_lr = DynamicReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,        # start with /2.5
        patience=2,
        mode='min',
        min_delta=0.0,
        cooldown=2,
        min_lr=1e-10,
        verbose=1,
        switch_epoch=20,
        switch_factor=custom_switch_factor  # after 20 epochs → divide by 10
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    # Save trained model - go one folder up
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    save_model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_single.h5")
    
    model.save(save_model_path)

    # Evaluate
    predictions_val = model.predict(X_val)
    predictions_val_original = invert_normalization(scaler, predictions_val[:, -1], n_features)
    y_val_original = invert_normalization(scaler, y_val[:, -1], n_features)

    mse = mean_squared_error(y_val_original, predictions_val_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_original, predictions_val_original)
    mape = mean_absolute_percentage_error(y_val_original, predictions_val_original) * 100
    normalized_rmse = rmse / (y_val_original.max() - y_val_original.min())

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "normalized_rmse": normalized_rmse
    }

    return save_model_path, metrics


# -------------------- CLI --------------------
if __name__ == "__main__":
    # Example usage: train BTC-USD for 60 min forecast
    symbol = "BTC-USD"
    print(f"Training GRU model for BTC 1 hour ahead forecast...")
    model_path, metrics = train_model()
    
    print(f"Model saved at {model_path}")
    print("Metrics:", metrics)
