import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
from scripts.predict import predict
from scripts.update_data import update_binance_ohlcv

# =================== CONFIGURACION ===================
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]
for tf in TIMEFRAMES:
    update_binance_ohlcv("PEPE/USDT", tf)

df_data = {tf: pd.read_csv(f"data/historical_data/PEPEUSDT_{tf}.csv") for tf in TIMEFRAMES}
for df in df_data.values():
    df["timestamp"] = pd.to_datetime(df["timestamp"])

model_paths = {tf: f"models/ppo_predictor_{tf}" for tf in TIMEFRAMES}
step_index = 0
window_minutes = 100

# =================== FUNCIONES ===================
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
    pattern = tuple(order)
    pattern_dict = {
        (0,1,2): 1,
        (0,2,1): 2,
        (1,0,2): 3,
        (1,2,0): 4,
        (2,0,1): 5,
        (2,1,0): 6
    }
    return pattern_dict.get(pattern, 0)

def update_plot():
    global step_index

    end_time = df_data["1m"]["timestamp"].iloc[-1] + timedelta(minutes=step_index)
    start_time = end_time - timedelta(minutes=window_minutes)

    fig.clear()
    ax = fig.add_subplot(111)
    frame = df_data["1m"]
    data = frame[(frame["timestamp"] >= start_time) & (frame["timestamp"] <= end_time)].copy()
    data["timestamp_local"] = data["timestamp"] - timedelta(hours=3)
    ax.plot(data["timestamp_local"], data["close"], label="Precio 1m", color="black")

    # Mostrar 60 minutos reales futuros si no estamos en el último minuto
    last_time = df_data["1m"]["timestamp"].iloc[-1]
    if end_time < last_time:
        future_data = frame[(frame["timestamp"] > end_time) & (frame["timestamp"] <= end_time + timedelta(minutes=60))].copy()
        if not future_data.empty:
            future_data["timestamp_local"] = future_data["timestamp"] - timedelta(hours=3)
            ax.plot(future_data["timestamp_local"], future_data["close"], label="Real futuro", color="green", linestyle="--")

    classification_text.delete("1.0", tk.END)
    local_time = end_time - timedelta(hours=3)
    classification_text.insert(tk.END, f"Predicciones al minuto: {local_time.strftime('%Y-%m-%d %H:%M')}\n\n")

    for tf in TIMEFRAMES:
        df_tf = df_data[tf]
        subset = df_tf[df_tf["timestamp"] <= end_time]
        input_df = prepare_data(subset)
        if len(input_df) >= 10:
            preds = predict(input_df, model_paths[tf], return_only=True)
            if len(preds) >= 3:
                pattern_id = classify(preds[:3])
                classification_text.insert(tk.END, f"{tf.upper()}: Patrón {pattern_id}\n")

    ax.set_title(f"PEPEUSDT 1m hasta {local_time.strftime('%Y-%m-%d %H:%M')}")
    ax.grid()
    canvas.draw()

# =================== NAVEGACION ===================
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

# =================== GUI ===================
root = tk.Tk()
root.title("Clasificador de Predicciones PEPEUSDT")
root.geometry("1200x600")

main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig = plt.Figure(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

info_frame = ttk.Frame(main_frame, width=200)
info_frame.pack(side=tk.RIGHT, fill=tk.Y)

classification_text = tk.Text(info_frame, wrap=tk.WORD, width=25, height=30)
classification_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.BOTTOM, pady=10)

btn_back = ttk.Button(btn_frame, text="<- Anterior", command=go_back)
btn_back.grid(row=0, column=0, padx=5)

btn_forward = ttk.Button(btn_frame, text="Siguiente ->", command=go_forward)
btn_forward.grid(row=0, column=1, padx=5)

btn_back_30 = ttk.Button(btn_frame, text="<- 30 min", command=back_30min)
btn_back_30.grid(row=0, column=2, padx=5)

btn_forward_30 = ttk.Button(btn_frame, text="30 min ->", command=forward_30min)
btn_forward_30.grid(row=0, column=3, padx=5)

btn_back_1h = ttk.Button(btn_frame, text="<- 1 h", command=back_1h)
btn_back_1h.grid(row=0, column=4, padx=5)

btn_forward_1h = ttk.Button(btn_frame, text="1 h ->", command=forward_1h)
btn_forward_1h.grid(row=0, column=5, padx=5)

btn_back_1d = ttk.Button(btn_frame, text="<- 1 d", command=back_1d)
btn_back_1d.grid(row=0, column=6, padx=5)

btn_forward_1d = ttk.Button(btn_frame, text="1 d ->", command=forward_1d)
btn_forward_1d.grid(row=0, column=7, padx=5)

update_plot()
root.mainloop()
