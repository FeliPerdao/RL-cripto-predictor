# Crypto RL Predictor

A Python project for training, predicting, and backtesting reinforcement learning (RL) models on cryptocurrency price data using Binance OHLCV data. The workflow automates data download, preprocessing, model training, prediction, and evaluation for multiple timeframes.

## Features

- **Automated Data Handling:** Downloads and updates OHLCV data for specified timeframes from Binance.
- **RL Model Training:** Trains PPO-based RL agents to predict price movements.
- **Prediction:** Predicts the next 3 candles (returns and prices) for each timeframe.
- **Backtesting:** (Optional) Evaluates model performance using random backtesting.

## Requirements

See and install requirements.txt

## Usage

Clone the repository and navigate to the project directory.

Run the main script:

python [main.py](VALID_FILE)
The script will:

Download (or update if existent) OHLCV data for PEPE/USDT across multiple timeframes.
Preprocess the data (calculate returns, format columns).
Train RL models (if not already trained).
Predict the next 3 candles for each timeframe.
Backtest the models.

Customization
Change Symbol: Edit the symbol in main.py (default: PEPE/USDT).
Add/Remove Timeframes: Modify the TIMEFRAMES list in main.py.

Logging
All logs are saved to logs/log.txt and also printed to the console.
