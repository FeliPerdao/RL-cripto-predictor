import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# === CARGA DE DATOS ===
df = pd.read_csv("data/historical_data/PEPEUSDT_1m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["close"] = df["close"].astype(float)

# Filtramos últimos 60 días
df = df[df["timestamp"] >= df["timestamp"].max() - timedelta(days=700)]

# === FUNCIONES DE DETECCIÓN ===
def detectar_patrones(df):
    patrones = []

    for i in range(30, len(df)-30):
        ventana = df.iloc[i-30:i+30].copy()
        precios = ventana["close"].values

        p0 = precios[29]
        p1 = precios[35]
        p2 = precios[59]

        # Cambio acumulado
        sube = (p1 - p0) / p0
        baja = (p2 - p1) / p1
        neto = (p2 - p0) / p0

        ts = df.iloc[i]["timestamp"]

        # Suba sostenida (> 5% en 30 min, sin baja importante)
        if neto > 0.05 and baja > -0.02:
            patrones.append((ts, "Suba sostenida"))

        # Suba falsa (> 5% seguido por baja > 4%)
        elif sube > 0.05 and baja < -0.04:
            patrones.append((ts, "Suba falsa"))

        # Desplome sostenido (< -5% sin rebote)
        elif neto < -0.05 and baja < -0.02:
            patrones.append((ts, "Desplome sostenido"))

        # Desplome falso (< -5% y rebote rápido)
        elif sube < -0.05 and baja > 0.04:
            patrones.append((ts, "Desplome falso"))

        # Toque en el fondo y rebote fuerte
        elif sube < -0.03 and baja > 0.06:
            patrones.append((ts, "Fondo con rebote"))

        # Pico y caída fuerte
        elif sube > 0.04 and baja < -0.06:
            patrones.append((ts, "Pico y caída"))

    return patrones

# === EJECUCIÓN ===
patrones_detectados = detectar_patrones(df)

# Mostrar y guardar resultados
if not os.path.exists("results"):
    os.makedirs("results")

result_path = "../results/patrones_detectados.csv"
df_result = pd.DataFrame(patrones_detectados, columns=["timestamp", "patron"])
df_result.to_csv(result_path, index=False)

print(f"Se detectaron {len(patrones_detectados)} patrones:")
for ts, patron in patrones_detectados:
    print(f"{ts} - {patron}")
