# Recurrent Neural Networks for Stock Price Prediction

Time series forecasting comparing RNN architectures (Simple RNN, LSTM, GRU) and ARIMA baseline on stock market data.

## Contents

- **RNN_LST_GRU.ipynb**: Complete implementation of RNN variants for stock price prediction
- **top50_adjclose_2010_2025.csv**: Historical adjusted close prices for top 50 US stocks (2010-2025)

## Overview

Comparative analysis of recurrent architectures for univariate time series prediction using Apple (AAPL) stock as the target. Data sourced from Yahoo Finance via `yfinance` library.

## Dataset

**Source**: Yahoo Finance  
**Period**: January 2010 - 2025  
**Features**: Adjusted close prices for 50 major US stocks  
**Companies**: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, V, and 41 others

## Models Implemented

### 1. ARIMA Baseline
- Auto-tuned ARIMA(p,d,q) using `pmdarima.auto_arima`
- Traditional statistical approach for comparison

### 2. Simple RNN
```
Input(60, 1) → SimpleRNN(128) → Dropout(0.2) →
Dense(64, LeakyReLU) → Dropout(0.2) →
Dense(32, LeakyReLU) → Dense(1)
```

### 3. LSTM (Long Short-Term Memory)
```
Input(60, 1) → LSTM(128) → Dropout(0.2) →
Dense(64, LeakyReLU) → Dropout(0.2) →
Dense(32, LeakyReLU) → Dense(1)
```

### 4. GRU (Gated Recurrent Unit)
```
Input(60, 1) → GRU(128) → Dropout(0.2) →
Dense(64, LeakyReLU) → Dropout(0.2) →
Dense(32, LeakyReLU) → Dense(1)
```

## Architecture Details

- **Lookback Window**: 60 days
- **Activation**: LeakyReLU for dense layers
- **Regularization**: Dropout (0.2)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Early Stopping**: patience=10, monitor='val_loss'

## Key Features

- **Proper Time Series Handling**: 
  - Sequential train/validation/test split (no shuffling)
  - MinMaxScaler normalization
  - Lookback window for sequential dependencies

- **Robust Evaluation**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

- **Visualization**:
  - Training/validation loss curves
  - Actual vs predicted prices comparison
  - Model performance metrics

## Typical Performance (AAPL)

Based on test set evaluation:

| Model       | MAE     | RMSE    | MAPE   |
|-------------|---------|---------|--------|
| Simple RNN  | $39.98  | $44.70  | 17.87% |
| LSTM        | $14.20  | $18.05  | 6.18%  |
| **GRU**     | **$7.11** | **$8.85** | **3.20%** |

*Note: Results vary depending on train/test split and market conditions*

## Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
pmdarima>=2.0.0
yfinance>=0.2.0
```

## Installation

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn statsmodels pmdarima yfinance
```

## Usage

### Running the Notebook
```python
jupyter notebook RNN_LST_GRU.ipynb
```

### Downloading Data (Optional)
```python
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOGL', ...]  # Top 50 stocks
data = yf.download(tickers, start='2010-01-01', end='2025-01-01')
adj_close = data['Adj Close']
adj_close.to_csv('top50_adjclose_2010_2025.csv')
```

## Data Preprocessing

1. **Normalization**: MinMaxScaler to [0, 1] range
2. **Sequence Creation**: Sliding window of 60 timesteps
3. **Split**: Chronological train/val/test (no shuffling)
4. **Reshaping**: (samples, timesteps, features) for RNN input

## Model Training

```python
# Common training configuration
epochs = 100
batch_size = 32
validation_split = 0.2
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

## Results Interpretation

- **GRU consistently outperforms** other architectures on this task
- LSTM performs well but slightly behind GRU in efficiency
- Simple RNN struggles with long-term dependencies
- ARIMA provides reasonable baseline but limited by linearity assumptions

## Why GRU Wins

1. **Simpler architecture**: Fewer parameters than LSTM (faster training)
2. **Efficient gating**: Update and reset gates capture dependencies effectively
3. **Better generalization**: Less prone to overfitting on financial data
4. **Computational efficiency**: ~30% faster than LSTM with similar performance

## Limitations

- **Single-step prediction**: Models predict next day only (not multi-horizon)
- **Univariate approach**: Only uses historical prices (no volume, indicators, or exogenous features)
- **No market regime detection**: Performance degrades during high volatility
- **Data leakage risk**: Ensure proper temporal ordering in production

## Future Improvements

- Multi-horizon forecasting (predict 5, 10, 30 days ahead)
- Multivariate models incorporating volume, technical indicators
- Attention mechanisms (Transformers, Temporal Fusion Transformer)
- Ensemble methods combining multiple architectures
- Walk-forward validation for realistic backtesting

## References

- LSTM: Hochreiter & Schmidhuber (1997)
- GRU: Cho et al. (2014)
- Yahoo Finance API: `yfinance` library

## License

MIT
