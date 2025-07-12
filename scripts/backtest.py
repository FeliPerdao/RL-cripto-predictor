import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env
import random

def backtest(data, model_path, steps=1, window_size=10, n_tests=100, test_split_only=True):
    """
    Evalúa el modelo sobre puntos aleatorios:
    - Si test_split_only=True: sólo dentro del 20% final del dataset
    - Si test_split_only=False: sobre todo el dataset

    Retorna el MSE promedio y la lista completa de errores.
    """
    data = data.copy().reset_index(drop=True) 
    
    # Validar columnas necesarias
    if "return" not in data.columns or "volume" not in data.columns:
        raise ValueError("❌ El dataset debe contener columnas 'return' y 'volume'")
    
    if test_split_only:
        test_start = int(len(data) * 0.8)
        data = data.iloc[test_start:].reset_index(drop=True)

    model = PPO.load(model_path)
    total_mse = []

    # Elegir posiciones aleatorias válidas
    max_start = len(data) - steps
    if max_start <= window_size:
        raise ValueError("⚠️ Muy pocos datos para testear.")
    
    positions = random.sample(range(window_size, max_start), min(n_tests, max_start - window_size))

    for pos in positions:
        env = CandlePredictionEnv(data, target_step=steps, window_size=window_size) #data, predict_steps=steps)
        env.position = pos
        obs = env._get_obs()
        
        # Se adapta a la cantidad de features
        if obs.shape != env.observation_space.shape:
            continue


        action, _ = model.predict(obs, deterministic=True)
        true_return = data.iloc[pos + steps - 1]["return"]

        mse = np.mean((action - true_return) ** 2)
        total_mse.append(mse)

    if not total_mse:
        raise ValueError("❌ No se pudo evaluar ningún punto válido.")

    avg_mse = np.mean(total_mse) #avg_error = total_error / n_tests
    return avg_mse, total_mse
