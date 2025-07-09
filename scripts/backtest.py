import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env
import random

def backtest(data, model_path, steps=3, n_tests=100, test_split_only=True):
    """
    Evalúa el modelo sobre puntos aleatorios:
    - Si test_split_only=True: sólo dentro del 20% final del dataset
    - Si test_split_only=False: sobre todo el dataset

    Retorna el MSE promedio y la lista completa de errores.
    """
    # Preprocesamiento: asegurar columnas necesarias
    data = data.copy().reset_index(drop=True) #data = data.copy()
    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)
    
    if test_split_only:
        test_start = int(len(data) * 0.8)
        data = data.iloc[test_start:].reset_index(drop=True)

    model = PPO.load(model_path)
    #total_error = 0
    total_mse = []

    # Elegir posiciones aleatorias válidas
    max_start = len(data) - steps - 10
    if max_start < 1:
        raise ValueError("⚠️ Muy pocos datos para testear.")
    positions = random.sample(range(10, max_start), min(n_tests, max_start - 10)) #positions = random.sample(range(10, max_start), n_tests)

    for pos in positions:
        env = CandlePredictionEnv(data, predict_steps=steps) #data, predict_steps=steps)
        env.position = pos
        obs = env._get_obs()
        
        if obs.shape != (20,):
            continue  # saltar observaciones inválidas

        action, _ = model.predict(obs, deterministic=True)
        true_returns = data.iloc[pos:pos + steps]["return"].values

        mse = np.mean((action - true_returns) ** 2)
        #total_error += mse
        total_mse.append(mse)

    if not total_mse:
        raise ValueError("❌ No se pudo evaluar ningún punto válido.")

    avg_mse = np.mean(total_mse) #avg_error = total_error / n_tests
    return avg_mse, total_mse
