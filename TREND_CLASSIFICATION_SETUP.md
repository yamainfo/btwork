# CryptoMamba Trend Classification Setup Guide

## Overview

This guide explains how to adapt the CryptoMamba codebase for binary trend classification using your new dataset with enhanced features.

## Dataset Requirements

Your dataset should have the following columns:
- `timestamp`: Unix timestamp
- `open`, `high`, `low`, `close`: OHLC price data
- `volume`: Trading volume
- `trend`: Binary target (1 for uptrend, 0 for downtrend)
- `ema_10`, `ema_30`, `ema_60`, `ema_120`: Exponential moving averages
- `ema_diff_60_10`, `ema_diff_120_60`: EMA differences
- `rsi_5`, `rsi_14`, `rsi_30`, `rsi_60`: RSI indicators

## Key Changes Made

### 1. Configuration Updates

#### Data Configuration (`configs/data_configs/mode_1.yaml`)
- Updated data path to point to your new dataset
- Added all additional features to `additional_features` list
- Set window size to 120 (configurable)

#### Training Configuration (`configs/training/cmamba_trend_classification.yaml`)
- New configuration specifically for trend classification
- Set `y_key: "trend"` for binary classification target
- Set `loss: "bce"` for binary cross-entropy loss
- Adjusted learning rate and hyperparameters for classification

#### Model Configuration (`configs/models/CryptoMamba/v1.yaml`)
- Updated `num_features: 16` (5 OHLCV + 10 additional + 1 volume)
- Set `hidden_dims: [120, 64, 32, 1]` for window size 120
- Changed target to classification module
- Set `loss: 'bce'` and `num_classes: 1`

### 2. New Classification Module

Created `pl_modules/cmamba_classification_module.py`:
- Extends base module for classification tasks
- Uses BCE loss for binary classification
- Includes accuracy and F1 score metrics
- Supports sigmoid activation for probability output

### 3. Enhanced Base Module

Updated `pl_modules/base_module.py`:
- Added support for both regression and classification modes
- Integrated BCE loss and classification metrics
- Dynamic monitoring based on task type (accuracy for classification, RMSE for regression)

### 4. Data Processing Updates

#### Data Transforms (`data_utils/data_transforms.py`)
- Updated to handle new feature names
- Improved feature processing for classification

#### Dataset Processing (`data_utils/dataset.py`)
- Added trend column processing
- Enhanced data merging for new features
- Support for binary trend targets

## Training Workflow

```mermaid
graph TD
    A[Load New Dataset] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Window Creation (120 timesteps)]
    D --> E[Binary Classification Model]
    E --> F[BCE Loss Calculation]
    F --> G[Accuracy/F1 Metrics]
    G --> H[Model Update]
    H --> I[Validation]
    I --> J[Checkpointing (Best Accuracy)]
    J --> K[Test Evaluation]
```

## How to Run Training

1. **Prepare your dataset**:
   ```bash
   # Place your CSV file in the data directory
   cp your_dataset.csv data/your_new_dataset.csv
   ```

2. **Update the data path** in `configs/data_configs/mode_1.yaml`:
   ```yaml
   data_path: "./data/your_new_dataset.csv"
   ```

3. **Run training**:
   ```bash
   python scripts/training.py \
     --config cmamba_trend_classification \
     --logdir ./logs \
     --batch_size 32 \
     --max_epochs 500 \
     --save_checkpoints
   ```

## Model Architecture

The classification model uses:
- **Input**: 16 features (OHLCV + technical indicators)
- **Window Size**: 120 timesteps
- **Architecture**: Mamba blocks with attention
- **Output**: Single value (logits for binary classification)
- **Activation**: Sigmoid for probability output
- **Loss**: Binary Cross-Entropy

## Monitoring and Metrics

During training, you'll see:
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Binary classification accuracy
- **F1 Score**: F1 score for trend prediction
- **Validation**: Same metrics on validation set

## Key Differences from Original

| Aspect | Original (Regression) | New (Classification) |
|--------|----------------------|---------------------|
| Target | Close price | Trend (0/1) |
| Loss | RMSE | BCE |
| Metrics | MSE, MAE, MAPE | Accuracy, F1 |
| Output | Continuous value | Probability |
| Window Size | 14 | 120 |
| Features | 5 (OHLCV) | 16 (OHLCV + technical) |

## Customization Options

1. **Window Size**: Modify `window_size` in config files
2. **Features**: Add/remove features in `additional_features`
3. **Model Size**: Adjust `hidden_dims` for different model capacities
4. **Learning Rate**: Tune `lr` and `lr_step_size` for your dataset
5. **Batch Size**: Adjust based on your GPU memory

## Troubleshooting

1. **Memory Issues**: Reduce batch size or window size
2. **Poor Performance**: Try different learning rates or model architectures
3. **Data Loading Errors**: Check CSV format and column names
4. **NaN Loss**: Check for invalid values in your dataset

## Next Steps

1. Run training with your dataset
2. Monitor validation metrics
3. Adjust hyperparameters based on performance
4. Evaluate on test set
5. Deploy for real-time predictions 