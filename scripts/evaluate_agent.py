import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv

def evaluate_agent(model_path, data, predict_steps=1, tf_name="", show_plot=False, save_dir="results"):
    """
    Evalúa el modelo sobre el 20% final del dataset, simulando paso a paso
    una predicción en tiempo real, graficando vs los valores reales.
    Guarda los valores de rewards en un dataframe
    """
    os.makedirs("results", exist_ok=True)
    data = data.copy().reset_index(drop=True)

    # Validar columnas necesarias
    if "return" not in data.columns or "volume" not in data.columns:
        raise ValueError("❌ El dataset debe contener columnas 'return' y 'volume'")

    test_start = int(len(data) * 0.8)
    test_data = data.iloc[test_start:].reset_index(drop=True)

    model = PPO.load(model_path)
    env = CandlePredictionEnv(test_data, target_step=predict_steps)

    predictions = []
    reals = []
    rewards = []
    positions = range(10, len(test_data) - predict_steps)

    for pos in positions:
        env.position = pos
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)

        true_return = test_data.iloc[pos + predict_steps - 1]["return"]
        predicted_return = action[0]
        reward = - (predicted_return - true_return) ** 2

        predictions.append(predicted_return)
        reals.append(true_return)
        rewards.append(reward)

    # Graficar
    plt.figure(figsize=(14, 6))
    plt.plot(predictions, label="Predicción", color="blue")
    plt.plot(reals, label="Real", color="green")
    reward_cumsum = np.cumsum(rewards[-50:])
    plt.plot(reward_cumsum, label="Reward acumulado", color="red", linestyle="--")

    plt.title(f"{tf_name} - Vela futura #{predict_steps}")
    plt.xlabel("Timestep")
    plt.ylabel("Variación / Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"eval_{tf_name}_step1.png")
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

    cumulative_reward = np.sum(rewards)
    print(f"\n🎯 Reward acumulado en {len(rewards)} pasos (Vela #{predict_steps}): {cumulative_reward:.6f}\n")

    reward_df = pd.DataFrame(
        [[cumulative_reward]],
        columns=[f"Vela_{predict_steps}"],
        index=[tf_name]
    )
    return reward_df