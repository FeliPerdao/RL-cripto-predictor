import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scripts.predict import predict
from scripts.update_data import update_binance_ohlcv

# ========== Actualización ==========
TIMEFRAMES = ["1m", "5m", "15m"]
for tf in TIMEFRAMES:
    update_binance_ohlcv("PEPE/USDT", tf)

# ========== Carga de datos ==========
df_1m = pd.read_csv("data/historical_data/PEPEUSDT_1m.csv")
df_5m = pd.read_csv("data/historical_data/PEPEUSDT_5m.csv")
df_15m = pd.read_csv("data/historical_data/PEPEUSDT_15m.csv")

for df in [df_1m, df_5m, df_15m]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

step_index = 0
window_minutes = 120
model_paths = {"5m": "models/ppo_predictor_5m", "15m": "models/ppo_predictor_15m"}

def prepare_data(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["return"] = df["close"].pct_change().fillna(0)
    df["volume"] = df["volume"].astype(float)
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_trend_up"] = (df["ema_9"] > df["ema_21"]).astype(int)
    return df[["return", "volume", "ema_9", "ema_21", "ema_trend_up"]]

def detect_pattern(preds):
    return len(preds) >= 3 and preds[0] < preds[1] > preds[2]

def update_plot():
    global step_index
    end_time = df_1m["timestamp"].iloc[-1] + timedelta(minutes=step_index)
    start_time = end_time - timedelta(minutes=window_minutes)

    data = df_1m[(df_1m["timestamp"] >= start_time) & (df_1m["timestamp"] <= end_time)].copy()
    data["timestamp_local"] = data["timestamp"] - timedelta(hours=3)

    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(data["timestamp_local"], data["close"], label="Precio 1m", color="black")

    for idx, row in data.iterrows():
        timestamp = row["timestamp"]
        x_time = row["timestamp_local"]
        close_price = row["close"]

        for tf, df_tf in zip(["5m", "15m"], [df_5m, df_15m]):
            subset = df_tf[df_tf["timestamp"] <= timestamp].copy()
            if len(subset) < 10:
                continue
            input_df = prepare_data(subset)
            preds = predict(input_df, model_paths[tf], return_only=True)
            if detect_pattern(preds[:3]):
                ax.plot(x_time, close_price, marker="o", color="green", markersize=6)
                break  # si ya marcó en una, no repite

    ax.set_title(f"Precio PEPEUSDT (1m) hasta {end_time.strftime('%Y-%m-%d %H:%M')}")
    ax.grid()
    canvas.draw()

def move(minutes):
    global step_index
    step_index += minutes
    if step_index > 0:
        step_index = 0
    update_plot()

def go_back(): move(-1)
def go_forward(): move(1)
def back_30min(): move(-30)
def forward_30min(): move(30)

def back_1h(): move(-60)
def forward_1h(): move(60)

def back_1d(): move(-1440)
def forward_1d(): move(1440)

# ========== GUI ==========
root = tk.Tk()
root.title("Detector de Picos - PEPEUSDT")
root.geometry("1000x600")

fig = plt.Figure(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

controls = [
    ("<- Anterior", go_back),
    ("Siguiente ->", go_forward),
    ("<- 30 min", back_30min),
    ("30 min ->", forward_30min),
    ("<- 1 h", back_1h),
    ("1 h ->", forward_1h),
    ("<- 1 d", back_1d),
    ("1 d ->", forward_1d),
]

for i, (label, cmd) in enumerate(controls):
    ttk.Button(btn_frame, text=label, command=cmd).grid(row=0, column=i, padx=4)

update_plot()
root.mainloop()
