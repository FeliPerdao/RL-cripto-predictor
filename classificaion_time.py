import pandas as pd
from datetime import datetime, timedelta
import os
from scripts.predict import predict
from scripts.update_data import update_binance_ohlcv

# ====== CONFIGURACIÓN ======
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]
SYMBOL = "PEPE/USDT"
RESULTS_PATH = "results/classification_log.csv"

# Actualizar y cargar datos
for tf in TIMEFRAMES:
    update_binance_ohlcv(SYMBOL, tf)

df_data = {tf: pd.read_csv(f"data/historical_data/PEPEUSDT_{tf}.csv") for tf in TIMEFRAMES}
for df in df_data.values():
    df["timestamp"] = pd.to_datetime(df["timestamp"])

model_paths = {tf: f"models/ppo_predictor_{tf}" for tf in TIMEFRAMES}

# ====== FUNCIONES ======
def prepare_data(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change().fillna(0)
    df["volume"] = df["volume"].astype(float)
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_trend_up"] = (df["ema_9"] > df["ema_21"]).astype(int)
    return df[["return", "volume", "ema_9", "ema_21", "ema_trend_up"]]

def classify(preds):
    order = sorted(range(len(preds)), key=lambda i: preds[i])
    pattern_dict = {
        (0, 1, 2): 1,
        (0, 2, 1): 2,
        (1, 0, 2): 3,
        (1, 2, 0): 4,
        (2, 0, 1): 5,
        (2, 1, 0): 6,
    }
    return pattern_dict.get(tuple(order), 0)

# ====== BUCLE PRINCIPAL ======
os.makedirs("results", exist_ok=True)
if not os.path.exists(RESULTS_PATH):
    # Crear CSV con encabezados si no existe
    empty = {
        "input_local_time": [],
        "prediction_utc_time": [],
    }
    for tf in TIMEFRAMES:
        empty[f"{tf}_pattern"] = []
        empty[f"{tf}_close"] = []
    pd.DataFrame(empty).to_csv(RESULTS_PATH, index=False)

print("🧠 Ingresá una hora local (formato: YYYY-MM-DD HH:MM), o escribí 'q' para salir.")

while True:
    user_input = input("\n⏰ Fecha y hora local: ").strip()
    if user_input.lower() == "q":
        print("👋 Saliste.")
        break

    try:
        local_time = datetime.strptime(user_input, "%Y-%m-%d %H:%M")
        utc_time = local_time + timedelta(hours=3)
    except ValueError:
        print("⚠️ Formato inválido. Usá: YYYY-MM-DD HH:MM")
        continue

    print(f"\n🧪 Clasificaciones para {user_input} (hora UTC: {utc_time.strftime('%Y-%m-%d %H:%M')})\n")

    row = {
        "input_local_time": user_input,
        "prediction_utc_time": utc_time.strftime('%Y-%m-%d %H:%M'),
    }

    for tf in TIMEFRAMES:
        df = df_data[tf]
        subset = df[df["timestamp"] <= utc_time]

        if len(subset) < 15:
            print(f"{tf.upper()}: ❌ No hay suficientes datos")
            row[f"{tf}_pattern"] = "No data"
            row[f"{tf}_close"] = "N/A"
            continue

        input_df = prepare_data(subset)
        preds = predict(input_df, model_paths[tf], return_only=True)

        try:
            close_price = subset.iloc[-1]["close"]
        except:
            close_price = "N/A"

        if len(preds) >= 3:
            pattern = classify(preds[:3])
            print(f"{tf.upper()}: Patrón {pattern} | Cierre: {close_price}")
            row[f"{tf}_pattern"] = pattern
            row[f"{tf}_close"] = close_price
        else:
            print(f"{tf.upper()}: ❌ No hay suficientes predicciones")
            row[f"{tf}_pattern"] = "No prediction"
            row[f"{tf}_close"] = close_price

    # Guardar
    df_result = pd.DataFrame([row])
    df_result.to_csv(RESULTS_PATH, mode="a", header=False, index=False)

    print("✅ Consulta registrada.")
