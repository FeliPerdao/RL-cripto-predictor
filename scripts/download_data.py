import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

def download_binance_ohlcv(symbol="PEPE/USDT", timeframe="1m", since_days=730):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=since_days)).isoformat())
    all_ohlcv = []
    limit = 1000

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    os.makedirs("data/historical_data", exist_ok=True)
    df.to_csv(f"data/historical_data/{symbol.replace('/', '')}_{timeframe}.csv", index=False)
    print(f"[\u2713] {symbol} - {timeframe} descargado.")