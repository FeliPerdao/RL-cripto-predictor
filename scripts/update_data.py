import ccxt
import pandas as pd
from datetime import datetime
import os

def update_binance_ohlcv(symbol="PEPE/USDT", timeframe="1m"):
    file_path = f"data/historical_data/{symbol.replace('/', '')}_{timeframe}.csv"
    exchange = ccxt.binance()

    # Si no existe, tiramos un warning y salimos
    if not os.path.exists(file_path):
        print(f"⚠️  Archivo {file_path} no existe. No se puede actualizar.")
        return

    # Leer el archivo actual
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_timestamp = df["timestamp"].max()

    # Binance trabaja en UTC
    since_ms = int(last_timestamp.timestamp() * 1000) + 60_000  # +1 minuto

    all_new = []
    limit = 1000

    while True:
        new_data = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not new_data:
            break
        df_new = pd.DataFrame(new_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
        all_new.append(df_new)
        since_ms = new_data[-1][0] + 1
        if len(new_data) < limit:
            break

    if all_new:
        df_all_new = pd.concat(all_new)
        df_combined = pd.concat([df, df_all_new])
        df_combined = df_combined.drop_duplicates(subset="timestamp").sort_values("timestamp")
        df_combined.to_csv(file_path, index=False)
        print(f"🆙 Datos de {timeframe} actualizados. Última vela: {df_combined['timestamp'].max()}")
    else:
        print(f"✅ {timeframe} ya está actualizado hasta {last_timestamp}")
