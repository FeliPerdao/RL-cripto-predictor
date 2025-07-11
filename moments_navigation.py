import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

# === CARGA DE DATOS ===
df = pd.read_csv("data/historical_data/PEPEUSDT_1m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["close"] = df["close"].astype(float)

# === CARGA DE PATRONES DETECTADOS ===
patrones = pd.read_csv("results/patrones_detectados.csv")
patrones["timestamp"] = pd.to_datetime(patrones["timestamp"])

# === GUI SETUP ===
root = tk.Tk()
root.title("Visualizador de Patrones PEPEUSDT")
root.geometry("1000x600")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig = plt.Figure(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

info_frame = ttk.Frame(main_frame, width=200)
info_frame.pack(side=tk.RIGHT, fill=tk.Y)

label_info = ttk.Label(info_frame, text="", font=("Arial", 12), wraplength=180, justify="center")
label_info.pack(pady=20)

btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

# === VARIABLES ===
index = len(patrones) - 1  # Comienza en el último registro

# === FUNCION PARA ACTUALIZAR GRAFICO ===
def update_plot():
    global index
    if index < 0:
        index = 0
    if index >= len(patrones):
        index = len(patrones) - 1

    ts = patrones.iloc[index]["timestamp"]
    clase = patrones.iloc[index]["patron"]
    label_info.config(text=f"{ts.strftime('%Y-%m-%d %H:%M:%S')}\n{clase}")

    inicio = ts - timedelta(minutes=50)
    fin = ts + timedelta(minutes=60)
    segmento = df[(df["timestamp"] >= inicio) & (df["timestamp"] <= fin)].copy()

    fig.clear()
    ax = fig.add_subplot(111)
    segmento["timestamp_local"] = segmento["timestamp"] - timedelta(hours=3)

    antes = segmento[segmento["timestamp"] <= ts]
    despues = segmento[segmento["timestamp"] > ts]

    ax.plot(antes["timestamp_local"], antes["close"], color="black", label="Antes")
    ax.plot(despues["timestamp_local"], despues["close"], color="green", label="Después")

    ax.axvline(ts - timedelta(hours=3), color="red", linestyle="--", label="Detección")
    ax.set_title(f"#{index+1} - PEPEUSDT 1m - Patrón: {clase}")
    ax.grid()
    ax.legend()
    canvas.draw()

# === FUNCIONES DE NAVEGACION ===
def siguiente():
    global index
    index += 1
    update_plot()

def anterior():
    global index
    index -= 1
    update_plot()

# === BOTONES ===
btn_prev = ttk.Button(btn_frame, text="<- Anterior", command=anterior)
btn_prev.grid(row=0, column=0, padx=10)

btn_next = ttk.Button(btn_frame, text="Siguiente ->", command=siguiente)
btn_next.grid(row=0, column=1, padx=10)

update_plot()
root.mainloop()
