import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv

def evaluate_agent_direction(model_path, data, predict_steps=1, tf_name="", show_plot=False, save_dir="results"):
    """
    Evalúa si el modelo acierta la dirección (sube/baja) para cada vela futura.
    Asigna +1 si acierta la dirección, -1 si falla.
    Genera gráficos y retorna un DataFrame con rewards acumulados.
    """
    os.makedirs("results", exist_ok=True)
    data = data.copy().reset_index(drop=True)

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
        true_returns = test_data.iloc[pos:pos + predict_steps]["return"].values

        if len(true_returns) <= predict_steps - 1:
            continue

        pred = action[0]
        real = true_returns[predict_steps - 1]

        reward = 1 if np.sign(pred) == np.sign(real) else -1

        predictions.append(pred)
        reals.append(real)
        rewards.append(reward)

    # Gráfico
    plt.figure(figsize=(14, 6))
    plt.plot(np.sign(predictions), label="Dirección Predicha", color="blue")
    plt.plot(np.sign(reals), label="Dirección Real", color="green")
    plt.plot(np.cumsum(rewards), label="Reward acumulado", color="red", linestyle="--")
    plt.title(f"{tf_name} - Dirección correcta (Vela +{predict_steps})")
    plt.xlabel("Timestep")
    plt.ylabel("Señal / Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"direction_{tf_name}_step1.png")
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

    cumulative_reward = np.sum(rewards)
    print(f"\n✨ Direcciones acertadas acumuladas (Vela +{predict_steps}): {cumulative_reward} de {len(rewards)}")
    accuracy = (cumulative_reward + len(rewards)) / (2 * len(rewards))
    print(f"📈 Precisión de dirección (Vela +{predict_steps}): {accuracy:.2%}")

    reward_df = pd.DataFrame(
        [[cumulative_reward]],
        columns=[f"Vela_{predict_steps}"],
        index=[tf_name]
    )
    return reward_df