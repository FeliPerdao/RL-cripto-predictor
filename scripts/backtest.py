import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env
import random

def backtest(data, model_path, steps=3, n_tests=100):
    # Preprocesamiento: asegurar columnas necesarias
    data = data.copy()
    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)

    model = PPO.load(model_path)
    total_error = 0
    total_mse = []

    # Elegir posiciones aleatorias válidas
    max_start = len(data) - steps - 10
    positions = random.sample(range(10, max_start), n_tests)

    for pos in positions:
        env = CandlePredictionEnv(data, predict_steps=steps)
        env.position = pos
        obs = env._get_obs()

        action, _ = model.predict(obs, deterministic=True)
        true_returns = data.iloc[pos:pos + steps]["return"].values

        mse = np.mean((action - true_returns) ** 2)
        total_error += mse
        total_mse.append(mse)

    avg_error = total_error / n_tests
    return avg_error, total_mse
