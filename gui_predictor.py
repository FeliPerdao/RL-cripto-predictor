import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scripts.predict import predict
import os

# ====================== CARGA DE DATOS ======================
df_1m = pd.read_csv("data/historical_data/PEPEUSDT_1m.csv")
df_3m = pd.read_csv("data/historical_data/PEPEUSDT_3m.csv")
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"])
df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])

# ===================== VARIABLES GLOBALES ====================
step_index = 0  # 0 es ahora, -1 es 1 min antes, +1 es 1 min después

# ===================== FUNCION DE PREDICCION ==================
def get_predictions(df, model_path):
    df = df.copy()
    df["return"] = df["close"].pct_change().fillna(0)  # ⬅️ línea crucial
    pred_returns = predict(df, model_path, return_only=True)
    return pred_returns


# ===================== ACTUALIZAR GRAFICO =====================
def update_plot():
    global step_index
    window_minutes = 50

    end_time = df_1m["timestamp"].iloc[-1] + timedelta(minutes=step_index)
    start_time = end_time - timedelta(minutes=window_minutes)

    data = df_1m[(df_1m["timestamp"] >= start_time) & (df_1m["timestamp"] <= end_time)].copy()

    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(data["timestamp"], data["close"], label="Precio real", color="black")

    if step_index <= 0:
        # Obtener último precio antes del punto actual
        last_10 = df_1m[df_1m["timestamp"] <= end_time].iloc[-10:].copy()
        last_close = last_10["close"].values[-1]

        # Cargar predicciones
        pred_1m = get_predictions(df_1m[df_1m["timestamp"] <= end_time], "models/ppo_predictor_1m")
        pred_3m = get_predictions(df_3m[df_3m["timestamp"] <= end_time], "models/ppo_predictor_3m")

        # === Predicción 1m ===
        future_times_1m = [end_time + timedelta(minutes=i + 1) for i in range(3)]
        prices_1m = [last_close * (1 + r) for r in pred_1m]
        ax.plot(future_times_1m, prices_1m, label="Predicción 1m", marker="o", linestyle="--", color="blue")

        # === Predicción 3m === (cada valor se repite 3 veces)
        future_times_3m = []
        prices_3m = []
        for i, r in enumerate(pred_3m):
            for j in range(3):
                future_times_3m.append(end_time + timedelta(minutes=i * 3 + j + 1))
                prices_3m.append(last_close * (1 + r))

        ax.plot(future_times_3m, prices_3m, label="Predicción 3m", marker="x", linestyle=":", color="orange")

        # === Agregar velas reales posteriores (si estamos en el pasado) ===
        real_future = df_1m[(df_1m["timestamp"] > end_time) & (df_1m["timestamp"] <= end_time + timedelta(minutes=9))]
        if not real_future.empty:
            ax.plot(real_future["timestamp"], real_future["close"], label="Real futuro", color="green", linewidth=2, alpha=0.6)

    ax.set_title(f"Precio PEPEUSDT hasta {end_time.strftime('%Y-%m-%d %H:%M')}")
    ax.legend()
    ax.grid()

    canvas.draw()


# ===================== FUNCIONES DE NAVEGACION =====================
def move(minutes):
    global step_index
    step_index += minutes
    # no dejamos ir hacia futuro más allá del último timestamp
    max_future = 0
    if step_index > max_future:
        step_index = max_future
    update_plot()

def go_back():
    move(-1)

def go_forward():
    move(1)

def back_30min():
    move(-30)

def forward_30min():
    move(30)

def back_1h():
    move(-60)

def forward_1h():
    move(60)

def back_1d():
    move(-1440)

def forward_1d():
    move(1440)
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
