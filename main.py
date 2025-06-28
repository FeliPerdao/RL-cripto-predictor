# main.py
import os
import pandas as pd
import sys
import logging
from scripts.download_data import download_binance_ohlcv
from scripts.agent import train_agent
from scripts.predict import predict
from scripts.backtest import backtest
from scripts.update_data import update_binance_ohlcv


# Configurar logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]

logging.info("=== INICIANDO PROCESO DE ENTRENAMIENTO Y PREDICCIÓN ===")

dataframes = {}
test_dataframes = {}

# Paso 1: Descargar datos si no existen y actualizar si existen
for tf in TIMEFRAMES:
    file_path = f"data/historical_data/PEPEUSDT_{tf}.csv"
    if not os.path.exists(file_path):
        logging.info(f"🔽 Descargando velas {tf}...")
        download_binance_ohlcv("PEPE/USDT", tf)
    else:
        logging.info(f"🔄 Archivo existente para {tf}, actualizando...")
        update_binance_ohlcv("PEPE/USDT", tf)

# Paso 2: Calcular variaciones porcentuales y guardar los dataframes
for tf in TIMEFRAMES:
    file_path = f"data/historical_data/PEPEUSDT_{tf}.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=["timestamp"], errors="ignore")
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change().fillna(0)
    df = df[["close", "return"]]  # también guardamos "close"
    dataframes[tf] = df
    logging.info(f"📈 Datos procesados para {tf}")

# Paso 3: Entrenar modelos (solo si no existen)
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    if os.path.exists(f"{model_path}.zip"):
        logging.info(f"🧠 Modelo ya existe para {tf}, salteando entrenamiento.")
        # Si ya existe, usamos el 20% final como test
        test_len = int(len(dataframes[tf]) * 0.2)
        test_dataframes[tf] = dataframes[tf].iloc[-test_len:].copy()
    else:
        logging.info(f"🧠 Entrenando modelo para {tf}...")
        test_df = train_agent(dataframes[tf], model_path)
        test_dataframes[tf] = test_df
        logging.info(f"✅ Modelo entrenado para {tf}")

# Paso 4: Predecir las próximas 3 velas
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    logging.info(f"\n🔮 Prediciendo próximas 3 velas (variación y precios) para {tf}:")
    predict(dataframes[tf], model_path, steps=3)

# Paso 5: Evaluar modelo con backtesting aleatorio (solo con test_df)
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    logging.info(f"\n📊 Backtest de modelo para {tf} (usando 20% test)...")
    avg_error, _ = backtest(test_dataframes[tf], model_path, steps=3, n_tests=100)
    logging.info(f"🔁 Error cuadrático medio promedio para {tf}: {avg_error:.6f}")
