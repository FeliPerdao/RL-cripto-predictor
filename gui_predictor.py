# Requiere: matplotlib, pandas, tkinter
# Supone que tenés las funciones `predict()` ya armadas y listas

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
    pred_returns = predict(df.copy(), model_path, return_only=True)  # modifica predict para retornar array
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
    ax.plot(data["timestamp"], data["close"], label="Precio real")

    # PREDICCIONES SOLO PARA EL AHORA O PASADO (NO FUTURO)
    if step_index <= 0:
        last_10 = df_1m[df_1m["timestamp"] <= end_time].iloc[-10:].copy()
        last_close = last_10["close"].values[-1]
        pred_1m = get_predictions(df_1m[df_1m["timestamp"] <= end_time], "models/ppo_predictor_1m")
        pred_3m = get_predictions(df_3m[df_3m["timestamp"] <= end_time], "models/ppo_predictor_3m")

        future_times = [end_time + timedelta(minutes=i+1) for i in range(3)]

        prices_1m = [last_close * (1 + r) for r in pred_1m]
        prices_3m = [last_close * (1 + r) for r in pred_3m]

        ax.plot(future_times, prices_1m, label="Predicción 1m", marker="o", linestyle="--")
        ax.plot(future_times, prices_3m, label="Predicción 3m", marker="x", linestyle=":")

    ax.set_title(f"Precio PEPEUSDT hasta {end_time.strftime('%Y-%m-%d %H:%M')}")
    ax.legend()
    ax.grid()

    canvas.draw()

# ===================== FUNCIONES DE NAVEGACION =====================
def go_back():
    global step_index
    step_index -= 1
    update_plot()

def go_forward():
    global step_index
    if step_index < 0:
        step_index += 1
    update_plot()

# ===================== GUI =====================
root = tk.Tk()
root.title("Predicciones PEPEUSDT")
root.geometry("1000x600")

fig = plt.Figure(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

btn_back = ttk.Button(btn_frame, text="<- Anterior", command=go_back)
btn_back.pack(side=tk.LEFT, padx=10)

btn_forward = ttk.Button(btn_frame, text="Siguiente ->", command=go_forward)
btn_forward.pack(side=tk.LEFT, padx=10)

update_plot()
root.mainloop()
