import os
import pandas as pd
import sys
import logging
from scripts.download_data import download_binance_ohlcv
from scripts.agent import train_agent
from scripts.predict import predict
from scripts.backtest import backtest
from scripts.update_data import update_binance_ohlcv
from scripts.evaluate_agent import evaluate_agent
from scripts.evaluate_agent_direction import evaluate_agent_direction

# Preguntar si se deben mostrar gráficos
show_graphs = input("¿Mostrar gráficos en pasos 6 y 7? (s/n): ").strip().lower() == 's'

# Configurar logging
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

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
    df["volume"] = df["volume"].astype(float)
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_trend_up"] = (df["ema_9"] > df["ema_21"]).astype(int) #para más polarización usar df["ema_trend_up"] = np.where(df["ema_9"] > df["ema_21"], 1, -1)
    
    # ------ FEATURES TO BE SENT TO THE AGENT -------
    df = df[["return", "volume", "ema_9", "ema_21", "ema_trend_up"]]
    
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

# Paso 5: Evaluar modelo con backtesting aleatorio de todo y de los úultimos 20%
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    
    #logging.info(f"\n📊 Backtest completo para {tf} (TODO el dataset)...")
    avg_all, _ = backtest(dataframes[tf], model_path, steps=3, n_tests=100, test_split_only=False)
    logging.info(f"🔁 MSE promedio (todo el dataset) para {tf}: {avg_all:.6f}")

    #logging.info(f"\n📊 Backtest parcial para {tf} (sólo 20% final)...")
    avg_last, _ = backtest(dataframes[tf], model_path, steps=3, n_tests=100, test_split_only=True)
    logging.info(f"🔁 MSE promedio (último 20%) para {tf}: {avg_last:.6f}\n")

# Paso 6: Evaluación visual con gráfico + reward acumulado
reward_matrix = []
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    reward_df = evaluate_agent(model_path, dataframes[tf], predict_steps=3, tf_name=tf, show_plot=show_graphs)
    reward_matrix.append(reward_df)

# Unir todo en un DataFrame y guardarlo
final_rewards_df = pd.concat(reward_matrix)
final_rewards_df.to_csv("results/rewards.csv")
print("\n📊 Tabla de Rewards acumulados por timeframe y vela:")
print(final_rewards_df)

# Paso 7: Evaluar si el modelo acierta la dirección de las velas futuras
direction_reward_df = pd.DataFrame()

for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}"
    logging.info(f"\n🎯 Evaluando dirección correcta para {tf}...")
    direction_df = evaluate_agent_direction(model_path, dataframes[tf], predict_steps=3, tf_name=tf, show_plot=show_graphs)
    direction_reward_df = pd.concat([direction_reward_df, direction_df])

# Guardar resultados
direction_reward_df.to_csv("results/direction_rewards.csv")
logging.info("\n📁 Rewards por dirección guardados en results/direction_rewards.csv")