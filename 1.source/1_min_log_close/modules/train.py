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
FEATURES = ['open', 'high', 'low', 'volume', 'low_1h', 'high_1h', 'log_close']
EPOCHS = 100
BATCH_SIZE = 128
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 120 mins sequence length
custom_switch_factor = 0.1

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
    return np.array(inv_data)


def build_model(n_features):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)))
    model.add(Dropout(0.1))
    model.add(GRU(50))
    model.add(Dropout(0.1))
    model.add(Dense(FORECAST_LENGTH * 3))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# -------------------- Training Function --------------------
def train_model():
    """
    Train a GRU model to predict log_close, low_1h, and high_1h for a 60-minute horizon.
    Saves model in models/GRU_BTC_60_1min.h5
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset_log_close.csv")

    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download data first.")

    # Load and validate data
    data = pd.read_csv(csv_path)
    missing = [col for col in FEATURES + ['close'] if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Determine sample counts for proper train/val split before scaling to avoid leakage
    total_rows = len(data)
    n_samples = total_rows - SEQ_LENGTH - FORECAST_LENGTH
    if n_samples <= 0:
        raise ValueError("Not enough rows to create any training samples. Reduce SEQ_LENGTH/FORECAST_LENGTH or add data.")

    train_samples = int(0.85 * n_samples)
    val_samples = int(0.10 * n_samples)
    # The index in the raw dataframe up to which training inputs end
    fit_end_index_exclusive = SEQ_LENGTH + train_samples

    # Fit scaler on training portion ONLY to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data.loc[:fit_end_index_exclusive - 1, FEATURES])

    # Transform the full dataset using the training-fitted scaler
    data_normalized = scaler.transform(data[FEATURES])

    # Build sequences on normalized data
    X, y = create_sequences_multi(data_normalized)

    # Flatten y for Dense output
    y_flat = y.reshape((y.shape[0], FORECAST_LENGTH * 3))

    # Train/validation split on sample axis
    n = len(X)
    train_split = train_samples
    val_split = train_split + val_samples if (train_split + val_samples) < n else int(0.95 * n)
    X_train, X_val = X[:train_split], X[train_split:val_split]
    y_train, y_val = y_flat[:train_split], y_flat[train_split:val_split]

    n_features = len(FEATURES)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
    
    model = build_model(n_features)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
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
        verbose=1,
        shuffle=False
    )

    # go one folder up
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    save_model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_1min.h5")
    
    model.save(save_model_path)

    # Evaluate on validation split
    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    predictions_val_original = invert_normalization_multi(scaler, predictions_val, n_features)

    # Separate predicted log returns
    pred_log_returns = predictions_val_original[:, :, 0]

    # Reconstruct actual close prices from log returns
    close_true = []
    close_pred = []

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

    mse_close = mean_squared_error(close_true[:, -1], close_pred[:, -1])
    rmse_close = np.sqrt(mse_close)
    mae_close = mean_absolute_error(close_true[:, -1], close_pred[:, -1])
    mape_close = mean_absolute_percentage_error(close_true[:, -1], close_pred[:, -1]) * 100
    normalized_rmse_close = rmse_close / (close_true.max() - close_true.min())

    metrics = {
        "mse": mse_close,
        "rmse": rmse_close,
        "mae": mae_close,
        "mape": mape_close,
        "normalized_rmse": normalized_rmse_close
}

    return save_model_path, metrics


# -------------------- CLI --------------------
if __name__ == "__main__":
    # Example usage: train BTC-USD for 60 min forecast
    symbol = "BTC-USD"
    print(f"Training GRU model for BTC 1 hour ahead forecast...")
    
    # Train the model
    model_path, metrics = train_model()
    
    print(f"Model saved at {model_path}")
    print("Metrics:", metrics)
