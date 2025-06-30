import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv

def evaluate_agent(model_path, data_path, predict_steps=3):
    # Cargar datos
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["return"] = df["close"].pct_change().fillna(0)
    df["volume"] = df["volume"].fillna(0)

    # Separar set de test
    test_df = df.iloc[int(len(df) * 0.8):].copy().reset_index(drop=True)

    # Cargar entorno
    env = CandlePredictionEnv(test_df, predict_steps=predict_steps)

    # Cargar modelo
    model = PPO.load(model_path)

    obs, _ = env.reset()
    predictions = []
    reales = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        predictions.append(action)
        futuros = test_df.iloc[env.position:env.position + predict_steps]["return"].values
        if len(futuros) == predict_steps:
            reales.append(futuros)

        if done:
            break

    # Convertir listas a arrays
    import numpy as np
    pred_array = np.array(predictions)
    real_array = np.array(reales)

    # === Plot comparativo ===
    plt.figure(figsize=(10, 5))
    for i in range(predict_steps):
        plt.plot(real_array[:, i], label=f"Real +{i+1}", linestyle="--", alpha=0.6)
        plt.plot(pred_array[:, i], label=f"Pred +{i+1}")

    plt.title("Evaluación del modelo PPO sobre datos de test")
    plt.xlabel("Timestep")
    plt.ylabel("Retorno (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ajustá esto a la ubicación real de tus archivos
    evaluate_agent("models/ppo_predictor_1m.zip", "data/historical_data/PEPEUSDT_1m.csv", predict_steps=3)
