import os, random
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.DynamicReduceLROnPlateau import DynamicReduceLROnPlateau

# -------------------- Performance/Determinism --------------------
# Remove explicit XLA disable and strict determinism for speed on GPU
# Seeds for reasonable reproducibility (not strict determinism)
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision for Ampere/Lovelace GPUs
mixed_precision.set_global_policy("mixed_float16")

# Enable dynamic GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)

# Enable XLA JIT for faster matmul/GRU kernels
try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled")
except Exception as e:
    print(f"Could not enable XLA JIT: {e}")

# -------------------- Config --------------------
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'low_1h', 'high_1h']
EPOCHS = 100
BATCH_SIZE = 256
FORECAST_LENGTH = 60 # 60 mins ahead forecast
SEQ_LENGTH = 120 # Last 120 mins sequence length
STEPS_PER_EXECUTION = 64
custom_switch_factor = 0.1

# -------------------- Helpers --------------------
def create_sequences_multi(data):
    """
    Create sequences for multiple target prediction: close, low_1h, high_1h
    """
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - FORECAST_LENGTH):
        X.append(data[i:(i + SEQ_LENGTH)])
        # Select columns: close=3, low=5, high=6
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


def build_model(n_features):
    model = Sequential()
    # Use cuDNN-accelerated GRU by setting reset_after=True and default activations
    model.add(GRU(128, return_sequences=True, input_shape=(SEQ_LENGTH, n_features), reset_after=True))
    model.add(Dropout(0.1))
    model.add(GRU(128, reset_after=True))
    model.add(Dropout(0.1))
    # Output in float32 for numerical stability with mixed precision
    model.add(Dense(FORECAST_LENGTH * 3, dtype='float32'))

    optimizer = Adam(learning_rate=3e-4, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        steps_per_execution=STEPS_PER_EXECUTION,
        run_eagerly=False,
    )
    return model

# -------------------- Training Function --------------------
def train_model():
    """
    Train a GRU model for a given crypto and forecast horizon.
    Saves model in models/GRU_BTC_60.h5
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    # csv_path = os.path.join(parent_dir, "data", "Bitcoin.csv")
    csv_path = os.path.join(parent_dir, "data", "Bitcoin_training_dataset.csv")

    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please download data first.")

    # Load and preprocess data
    data = pd.read_csv(csv_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data[FEATURES])

    X, y = create_sequences_multi(data_normalized)

    # Flatten y for Dense output
    y_flat = y.reshape((y.shape[0], FORECAST_LENGTH * 3)).astype(np.float32)

    # Train/validation split
    n = len(X)
    train_split = int(0.9 * n)
    val_split = int(0.97 * n)
    X_train, X_val = X[:train_split], X[train_split:val_split]
    y_train, y_val = y_flat[:train_split], y_flat[train_split:val_split]

    n_features = len(FEATURES)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features)).astype(np.float32)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features)).astype(np.float32)
    
    # Optimized tf.data pipelines
    options = tf.data.Options()
    options.experimental_deterministic = False

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds
        .cache()
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_ds = train_ds.with_options(options)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = (
        val_ds
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.with_options(options)

    with tf.device('/GPU:0'):
        model = build_model(n_features)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    # Learning rate manipulation
    reduce_lr = DynamicReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # start with 1/2
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
        train_ds, 
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    # go one folder up
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    save_model_path = os.path.join(parent_dir, "models", "GRU_BTC_60_multi.h5")
    
    model.save(save_model_path)

    # Evaluate
    predictions_val = model.predict(X_val, batch_size=BATCH_SIZE)
    predictions_val = predictions_val.reshape((-1, FORECAST_LENGTH, 3))

    predictions_val_original = invert_normalization_multi(scaler, predictions_val, n_features)
    y_val_original = invert_normalization_multi(scaler, y_val.reshape((-1, FORECAST_LENGTH, 3)), n_features)

    # Compute RMSE for close only as example
    close_pred = predictions_val_original[:, -1, 0]
    close_true = y_val_original[:, -1, 0]
    mse_close = mean_squared_error(close_true, close_pred)
    rmse_close = np.sqrt(mse_close)
    mae_close = mean_absolute_error(close_true, close_pred)
    mape_close = mean_absolute_percentage_error(close_true, close_pred) * 100
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
    
    # Check GPU
    print("TensorFlow version:", tf.__version__)
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Train the model
    model_path, metrics = train_model()
    
    print(f"Model saved at {model_path}")
    print("Metrics:", metrics)
