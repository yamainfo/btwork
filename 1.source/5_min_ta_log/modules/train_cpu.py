import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.DynamicReduceLROnPlateau import DynamicReduceLROnPlateau

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

EPOCHS = 100
BATCH_SIZE = 128
FORECAST_LENGTH = 12  # 5 * 12 = 60 mins ahead
SEQ_LENGTH = 60       # last 60*5min = 300 mins
custom_switch_factor = 0.1

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


def invert_normalization_multi_diff(scaler, data, n_features):
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


def build_model(n_features):
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)))
    model.add(GRU(50))
    model.add(Dense(FORECAST_LENGTH * 3))
    #model.compile(optimizer="adam", loss="mean_square_error")
    model.compile(optimizer="adam", loss=Huber(delta=1.0))
    return model


# -------------------- Training Function --------------------
def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download data first.")

    data = pd.read_csv(csv_path)

    # Log-transform volume
    data['log_volume'] = np.log1p(data['volume'])

    # Select features and normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences_multi(data_normalized)
    y_flat = y.reshape((y.shape[0], FORECAST_LENGTH * 3))

    # Train/validation split
    n = len(X)
    train_split = int(0.90 * n)
    val_split = int(0.99 * n)
    X_train, X_val = X[:train_split], X[train_split:val_split]
    y_train, y_val = y_flat[:train_split], y_flat[train_split:val_split]

    n_features = len(FEATURES)
    model = build_model(n_features)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = DynamicReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        mode='min',
        min_delta=0.0,
        cooldown=2,
        min_lr=1e-10,
        verbose=1,
        switch_epoch=20,
        switch_factor=custom_switch_factor
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    save_model_path = os.path.join(parent_dir, "models", "BTC_5min_ta_complex.h5")
    model.save(save_model_path)

    # -------------------- Evaluation --------------------
    predictions_val = model.predict(X_val)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    # Inverse scale dlog predictions
    predictions_val_log = invert_normalization_multi_diff(scaler, predictions_val, n_features)
    y_val_log = invert_normalization_multi_diff(scaler, y_val.reshape((-1, FORECAST_LENGTH, 3)), n_features)

    # Convert log prices to actual BTC price in USD
    pred_close = np.exp(predictions_val_log[:, -1, 0])
    true_close = np.exp(y_val_log[:, -1, 0])

    # Compute evaluation metrics in USD
    mse_close = mean_squared_error(true_close, pred_close)
    rmse_close = np.sqrt(mse_close)
    mae_close = mean_absolute_error(true_close, pred_close)
    mape_close = mean_absolute_percentage_error(true_close, pred_close) * 100
    normalized_rmse_close = rmse_close / (true_close.max() - true_close.min())

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
    print("Training GRU model for BTC 1 hour ahead forecast...")
    model_path, metrics = train_model()
    print(f"Model saved at {model_path}")
    print("Metrics (USD):", metrics)
