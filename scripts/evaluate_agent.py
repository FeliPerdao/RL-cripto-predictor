import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
import time


def evaluate_agent(model_path, data, predict_steps=3, tf_name=""):
    """
    Evalúa el modelo sobre el 20% final del dataset, simulando paso a paso
    una predicción en tiempo real, graficando vs los valores reales.
    Guarda los valores de rewards en un dataframe
    """
    data = data.copy().reset_index(drop=True)
    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)

    test_start = int(len(data) * 0.8)
    test_data = data.iloc[test_start:].reset_index(drop=True)

    model = PPO.load(model_path)
    env = CandlePredictionEnv(test_data, predict_steps=predict_steps)

    all_rewards = []
    #rewards_dict = {}

    for i in range(predict_steps):
        predictions = []
        reals = []
        rewards = []
        positions = range(10, len(test_data) - predict_steps)

        for pos in positions:
            env.position = pos
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)

            true_returns = test_data.iloc[pos:pos + predict_steps]["return"].values

            if len(true_returns) <= i:
                continue

            reward = - (action[i] - true_returns[i]) ** 2

            predictions.append(action[i])
            reals.append(true_returns[i])
            rewards.append(reward)

        # Graficar solo las últimas 50 predicciones
        plt.figure(figsize=(14, 6))
        plt.plot(predictions[-50:], label="Predicción", color="blue")
        plt.plot(reals[-50:], label="Real", color="green")

        # Agregar evolución de reward acumulado
        reward_cumsum = np.cumsum(rewards[-50:])
        plt.plot(reward_cumsum, label="Reward acumulado", color="red", linestyle="--")

        plt.title(f"{tf_name} - Vela futura #{i+1}")
        plt.xlabel("Timestep")
        plt.ylabel("Variación / Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        cumulative_reward = np.sum(rewards)
        print(f"\n🎯 Reward acumulado en {len(rewards)} pasos (Vela #{i+1}): {cumulative_reward:.6f}\n")

        all_rewards.append(cumulative_reward)

    # Retornar DataFrame con rewards por vela
    reward_df = pd.DataFrame([all_rewards], columns=[f"Vela_{i+1}" for i in range(predict_steps)], index=[tf_name])
    return reward_df