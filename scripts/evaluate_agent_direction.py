import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv

def evaluate_agent_direction(model_path, data, predict_steps=3, tf_name="", show_plot=False):
    """
    Evalúa si el modelo acierta la dirección (sube/baja) para cada vela futura.
    Asigna +1 si acierta la dirección, -1 si falla.
    Genera gráficos y retorna un DataFrame con rewards acumulados.
    """
    os.makedirs("results", exist_ok=True)
    data = data.copy().reset_index(drop=True)

    # Validación de columnas necesarias
    if "return" not in data.columns or "volume" not in data.columns:
        raise ValueError("❌ El dataset debe contener columnas 'return' y 'volume'")
    
    test_start = int(len(data) * 0.8)
    test_data = data.iloc[test_start:].reset_index(drop=True)

    model = PPO.load(model_path)
    env = CandlePredictionEnv(test_data, predict_steps=predict_steps)

    all_rewards = []

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

            pred = action[i]
            real = true_returns[i]

            reward = 1 if np.sign(pred) == np.sign(real) else -1
            #reward = 2 if np.sign(pred) == np.sign(real) else -1 #posible tunning para exloracion

            predictions.append(pred)
            reals.append(real)
            rewards.append(reward)

        # Gráfico
        plt.figure(figsize=(14, 6))
        plt.plot(np.sign(predictions[:]), label="Dirección Predicha", color="blue")
        plt.plot(np.sign(reals[:]), label="Dirección Real", color="green")
        plt.plot(np.cumsum(rewards[:]), label="Reward acumulado", color="red", linestyle="--")
        plt.title(f"{tf_name} - Dirección correcta Vela #{i+1}")
        plt.xlabel("Timestep")
        plt.ylabel("Señal / Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = f"results/direction_{tf_name}_step{i+1}.png"
        plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()

        cumulative_reward = np.sum(rewards)
        print(f"\n✨ Direcciones acertadas acumuladas (Vela #{i+1}): {cumulative_reward} de {len(rewards)}")
        accuracy = (cumulative_reward + len(rewards)) / (2 * len(rewards))
        print(f"📈 Precisión de dirección (Vela #{i+1}): {accuracy:.2%}")
        all_rewards.append(cumulative_reward)

    reward_df = pd.DataFrame(
        [all_rewards],
        columns=[f"Vela_{i+1}" for i in range(predict_steps)],
        index=[tf_name]
    )
    return reward_df