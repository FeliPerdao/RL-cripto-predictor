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
import re
from datetime import datetime

# === MODEL SETTINGS ===
target_step = int(input("¿Qué vela futura querés predecir? (1, 2, 3...): "))
window_size = int(input("¿Con cuántas velas anteriores querés entrenar? (por ejemplo, 10): "))

# === CREAR CARPETA DE SALIDA ===
today = datetime.now().strftime("%y-%m-%d")
base_dir = "results/runs"
os.makedirs(base_dir, exist_ok=True)

# Buscar el número siguiente
existing = sorted([
    d for d in os.listdir(base_dir) 
    if os.path.isdir(os.path.join(base_dir, d)) and d.split(" - ")[0].isdigit()
])
next_num = int(existing[-1].split(" - ")[0]) + 1 if existing else 1

# Crear el nombre de la carpeta
run_name = f"{next_num} - {today} - Vela {target_step} - Window {window_size}"
run_dir = os.path.join(base_dir, run_name)
os.makedirs(run_dir, exist_ok=True)

logging.info(f"\n📂 Carpeta de corrida creada: {run_dir}")

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
    model_path = f"models/ppo_predictor_{tf}_step{target_step}_win{window_size}"
    if os.path.exists(f"{model_path}.zip"):
        logging.info(f"🧠 Modelo ya existe para {tf} (step {target_step}, window {window_size}), salteando entrenamiento.")
        test_len = int(len(dataframes[tf]) * 0.2)
        test_dataframes[tf] = dataframes[tf].iloc[-test_len:].copy()
    else:
        logging.info(f"🧠 Entrenando modelo para {tf} (step {target_step}, window {window_size})...")
        test_df = train_agent(dataframes[tf], model_path, predict_steps=target_step, window_size=window_size)
        test_dataframes[tf] = test_df
        logging.info(f"✅ Modelo entrenado para {tf}")

# Paso 4: Predecir las próxima enésima vela
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}_step{target_step}_win{window_size}"
    logging.info(f"\n🔮 Prediciendo vela {target_step} para {tf}:")
    predict(dataframes[tf], model_path, steps=target_step, window_size=window_size)

# Paso 5: Evaluar modelo con backtesting aleatorio de todo y de los úultimos 20%
for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}_step{target_step}_win{window_size}"
    
    avg_all, _ = backtest(dataframes[tf], model_path, steps=target_step, window_size=window_size, n_tests=100, test_split_only=False)
    logging.info(f"🔁 MSE promedio (todo el dataset) para {tf}: {avg_all:.6f}")
    
    avg_last, _ = backtest(dataframes[tf], model_path, steps=target_step, window_size=window_size, n_tests=100, test_split_only=True)
    logging.info(f"🔁 MSE promedio (último 20%) para {tf}: {avg_last:.6f}\n")

# Paso 6: Evaluación visual con reward acumulado
rewards_file = "results/rewards.csv"
if os.path.exists(rewards_file):
    master_df = pd.read_csv(rewards_file, index_col=0)
else:
    master_df = pd.DataFrame()

for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}_step{target_step}_win{window_size}"
    reward_df = evaluate_agent(model_path, dataframes[tf], predict_steps=target_step, tf_name=tf, show_plot=show_graphs, save_dir=run_dir)

    # Nueva columna: por ejemplo, Vela_1-w10
    new_col = f"Vela_{target_step}-w{window_size}"
    original_col = f"Vela_{target_step}"
    if original_col in reward_df.columns:
        reward_df = reward_df.rename(columns={original_col: new_col})
    else:
        print(f"⚠️ Warning: columna {original_col} no encontrada en reward_df")

    # Si ya existe, reemplaza el valor. Si no, lo agrega.
    if tf in master_df.index:
        master_df.loc[tf, new_col] = reward_df.loc[tf, new_col]
    else:
        master_df.loc[tf] = reward_df.loc[tf]

# Ordenar columnas: primero por vela (número), luego por window_size (w) crecientes
def ordenar_columnas(cols):
    def extraer_valores(c):
        match = re.match(r"Vela_(\d+)-w(\d+)", c)
        return (int(match.group(1)), int(match.group(2))) if match else (999, 999)
    return sorted(cols, key=extraer_valores)

master_df = master_df[ordenar_columnas(master_df.columns)]

# Guardar
master_df.to_csv(rewards_file)

print("\n📊 Tabla de Rewards acumulados por timeframe y vela-window:")
print(master_df)

# Paso 7: Evaluar si el modelo acierta la dirección de las velas futuras
# Cargar archivo existente si existe
direction_file = "results/direction_rewards.csv"
if os.path.exists(direction_file):
    direction_master_df = pd.read_csv(direction_file, index_col=0)
else:
    direction_master_df = pd.DataFrame()

for tf in TIMEFRAMES:
    model_path = f"models/ppo_predictor_{tf}_step{target_step}_win{window_size}"
    logging.info(f"\n🎯 Evaluando dirección correcta para {tf}...")
    dir_df = evaluate_agent_direction(model_path, dataframes[tf], predict_steps=target_step, tf_name=tf, show_plot=show_graphs, save_dir=run_dir)

    # Nueva columna: por ejemplo, Vela_1-w10
    new_col = f"Vela_{target_step}-w{window_size}"
    dir_df = dir_df.rename(columns={f"Vela_{target_step}": new_col})

    # Actualizar o agregar al master
    if tf in direction_master_df.index:
        direction_master_df.loc[tf, new_col] = dir_df.loc[tf, new_col]
    else:
        # Si no hay columnas aún, asegurarse de crearlas
        if direction_master_df.empty:
            direction_master_df = pd.DataFrame(columns=dir_df.columns)
        direction_master_df.loc[tf] = dir_df.loc[tf]

# Ordenar columnas
def ordenar_columnas(cols):
    def extraer(c):
        m = re.match(r"Vela_(\d+)-w(\d+)", c)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
    return sorted(cols, key=extraer)

direction_master_df = direction_master_df[ordenar_columnas(direction_master_df.columns)]

# Guardar
direction_master_df.to_csv(direction_file)

logging.info("\n📁 Rewards por dirección guardados en results/direction_rewards.csv")
print("\n🎯 Tabla de Dirección acumulada por timeframe y vela-window:")
print(direction_master_df)