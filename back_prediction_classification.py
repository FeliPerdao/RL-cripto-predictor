import pandas as pd
import os
from datetime import datetime, timedelta
from scripts.predict import predict

# ================= CONFIG ====================
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]
MODEL_PATHS = {tf: f"models/ppo_predictor_{tf}" for tf in TIMEFRAMES}
DATA_PATHS = {tf: f"data/historical_data/PEPEUSDT_{tf}.csv" for tf in TIMEFRAMES}

# ============== FUNCIONES =====================
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

# ============= CARGA DE DATOS ==================
print("Cargando datos...")
patrones_df = pd.read_csv("results/patrones_detectados.csv")
patrones_df["timestamp"] = pd.to_datetime(patrones_df["timestamp"])

df_data = {}
for tf in TIMEFRAMES:
    df = pd.read_csv(DATA_PATHS[tf])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["close"] = df["close"].astype(float)
    df_data[tf] = df

# ============ PROCESAR PREDICCIONES ============
results = []
total = len(patrones_df)

print(f"Analizando {total} registros...\n")

for i, row in patrones_df.iterrows():
    ts = row["timestamp"]
    patron = row["patron"]

    # Mostrar progreso
    pct = (i + 1) / total * 100
    print(f"[{i+1}/{total}] {ts} - {patron} - {pct:.2f}% completado")

    entry = {
        "timestamp": ts,
        "patron": patron,
    }

    # Precio de cierre en 1m
    df_1m = df_data["1m"]
    row_close = df_1m[df_1m["timestamp"] == ts]
    entry["close"] = row_close.iloc[0]["close"] if not row_close.empty else None

    # Clasificación por timeframe
    for tf in TIMEFRAMES:
        df = df_data[tf]
        df_tf = df[df["timestamp"] <= ts]
        input_df = prepare_data(df_tf)

        if len(input_df) < 10:
            entry[f"pattern_{tf}"] = 0
            for j in range(1, 4):
                entry[f"{tf}_step{j}"] = None
            continue

        preds = predict(input_df, MODEL_PATHS[tf], return_only=True)

        if len(preds) >= 3:
            entry[f"pattern_{tf}"] = classify(preds[:3])
            for j in range(3):
                entry[f"{tf}_step{j+1}"] = preds[j]
        else:
            entry[f"pattern_{tf}"] = 0
            for j in range(1, 4):
                entry[f"{tf}_step{j}"] = None

    results.append(entry)

# ============= GUARDAR RESULTADO ================
output_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
output_df.to_csv("results/patrones_clasificados1.csv", index=False)
print("\n✅ Clasificación completada. Archivo guardado en results/patrones_clasificados.csv")
