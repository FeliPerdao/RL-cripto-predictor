import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scripts.predict import predict
import os
from scripts.update_data import update_binance_ohlcv

# ================== ACTUALIZACIÓN DE DATOS ==================
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h"] #, "1d"
for tf in TIMEFRAMES:
    update_binance_ohlcv("PEPE/USDT", tf)


# ====================== CARGA DE DATOS ======================
df_1m = pd.read_csv("data/historical_data/PEPEUSDT_1m.csv")
df_3m = pd.read_csv("data/historical_data/PEPEUSDT_3m.csv")
df_5m = pd.read_csv("data/historical_data/PEPEUSDT_5m.csv")
df_15m = pd.read_csv("data/historical_data/PEPEUSDT_15m.csv")
df_1h = pd.read_csv("data/historical_data/PEPEUSDT_1h.csv")
df_1d = pd.read_csv("data/historical_data/PEPEUSDT_1d.csv")

for df in [df_1m, df_3m, df_5m, df_15m, df_1h]: #, df_1d
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# ===================== VARIABLES GLOBALES ====================
step_index = 0  # 0 es ahora, -1 es 1 min antes, +1 es 1 min después

# ===================== FUNCION DE PREDICCION ==================
def get_predictions(df, model_path):
    df = df.copy()
    df["return"] = df["close"].pct_change().fillna(0) 
    pred_returns = predict(df, model_path, return_only=True)
    return pred_returns


# ===================== ACTUALIZAR GRAFICO =====================
def update_plot():
    global step_index
    window_minutes = 50

    end_time = df_1m["timestamp"].iloc[-1] + timedelta(minutes=step_index)
    start_time = end_time - timedelta(minutes=window_minutes)

    data = df_1m[(df_1m["timestamp"] >= start_time) & (df_1m["timestamp"] <= end_time)].copy()
    data["timestamp_local"] = data["timestamp"] - timedelta(hours=3)


    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(data["timestamp_local"], data["close"], label="Precio real", color="black")

    if step_index <= 0:
        # Obtener último precio antes del punto actual
        last_10 = df_1m[df_1m["timestamp"] <= end_time].iloc[-10:].copy()
        last_close = last_10["close"].values[-1]

        # Cargar predicciones
        preds = {
            "1m": (get_predictions(df_1m[df_1m["timestamp"] <= end_time], "models/ppo_predictor_1m"), 1, "blue", "o", "--"),
            "3m": (get_predictions(df_3m[df_3m["timestamp"] <= end_time], "models/ppo_predictor_3m"), 3, "orange", "x", ":"),
            "5m": (get_predictions(df_5m[df_5m["timestamp"] <= end_time], "models/ppo_predictor_5m"), 5, "red", "^", "-"),
            "15m": (get_predictions(df_15m[df_15m["timestamp"] <= end_time], "models/ppo_predictor_15m"), 10, "purple", "s", "-"),
            "1h": (get_predictions(df_1h[df_1h["timestamp"] <= end_time], "models/ppo_predictor_1h"), 20, "green", "d", "-"),
            #"1d": (get_predictions(df_1d[df_1d["timestamp"] <= end_time], "models/ppo_predictor_1d"), 1440, "brown", "P", "-"),
        }
        
        for label, (preds_arr, interval, color, marker, linestyle) in preds.items():
            future_times = []
            prices = []
            for i, r in enumerate(preds_arr):
                rep = 1 if label == "1d" else interval
                for j in range(rep):
                    future_times.append(end_time + timedelta(minutes=i * interval + j + 1))
                    prices.append(last_close * (1 + r))
            future_times = [t - timedelta(hours=3) for t in future_times]
            ax.plot(future_times, prices, label=f"Predicción {label}", marker=marker, linestyle=linestyle, color=color)

        real_future = df_1m[(df_1m["timestamp"] > end_time) & (df_1m["timestamp"] <= end_time + timedelta(minutes=90))]
        if not real_future.empty:
            real_future["timestamp_local"] = real_future["timestamp"] - timedelta(hours=3)
            ax.plot(real_future["timestamp_local"], real_future["close"], label="Real futuro", color="green", linewidth=2, alpha=0.6)

    ax.set_title(f"Precio PEPEUSDT hasta {end_time.strftime('%Y-%m-%d %H:%M')}")
    ax.legend()
    ax.grid()

    canvas.draw()


# ===================== FUNCIONES DE NAVEGACION =====================
def move(minutes):
    global step_index
    step_index += minutes
    max_future = 0 # no dejamos ir hacia futuro más allá del último timestamp
    if step_index > max_future:
        step_index = max_future
    update_plot()

def go_back(): move(-1)
def go_forward(): move(1)
def back_30min(): move(-30)
def forward_30min(): move(30)
def back_1h(): move(-60)
def forward_1h(): move(60)
def back_1d(): move(-1440)
def forward_1d(): move(1440)
# ===================== GUI =====================
root = tk.Tk()
root.title("Predicciones PEPEUSDT")
root.geometry("1000x600")

fig = plt.Figure(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

# Botones 1 minuto
btn_back = ttk.Button(btn_frame, text="<- Anterior", command=go_back)
btn_back.grid(row=0, column=0, padx=5)

btn_forward = ttk.Button(btn_frame, text="Siguiente ->", command=go_forward)
btn_forward.grid(row=0, column=1, padx=5)

# Botones 30 minutos
btn_back_30 = ttk.Button(btn_frame, text="<- 30 min", command=back_30min)
btn_back_30.grid(row=0, column=2, padx=5)

btn_forward_30 = ttk.Button(btn_frame, text="30 min ->", command=forward_30min)
btn_forward_30.grid(row=0, column=3, padx=5)

# Botones 1 hora
btn_back_1h = ttk.Button(btn_frame, text="<- 1 h", command=back_1h)
btn_back_1h.grid(row=0, column=4, padx=5)

btn_forward_1h = ttk.Button(btn_frame, text="1 h ->", command=forward_1h)
btn_forward_1h.grid(row=0, column=5, padx=5)

# Botones 1 día
btn_back_1d = ttk.Button(btn_frame, text="<- 1 d", command=back_1d)
btn_back_1d.grid(row=0, column=6, padx=5)

btn_forward_1d = ttk.Button(btn_frame, text="1 d ->", command=forward_1d)
btn_forward_1d.grid(row=0, column=7, padx=5)

update_plot()
root.mainloop()
