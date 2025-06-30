import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env
import random

def backtest(data, model_path, steps=3, n_tests=100, verbose=False):
    """
    Evalúa el modelo PPO sobre datos históricos, comparando predicciones con retornos reales.
    
    Parámetros:
    - data: DataFrame con columnas 'close' y 'volume'
    - model_path: path al modelo PPO entrenado
    - steps: cantidad de velas a predecir (horizonte)
    - n_tests: cantidad de predicciones aleatorias a evaluar
    - verbose: si True, imprime resultados individuales
    
    Retorna:
    - error promedio (MSE)
    - lista con errores individuales
    """
    # Preprocesamiento: asegurar columnas necesarias
    data = data.copy()
    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)

    model = PPO.load(model_path)
    total_mse = []

    # Elegir posiciones aleatorias válidas
    max_start = len(data) - steps - 10
    positions = random.sample(range(10, max_start), n_tests)
    
    env = CandlePredictionEnv(data, predict_steps=steps)

    for pos in positions:
        env.position = pos
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        true_returns = data.iloc[pos:pos + steps]["return"].values
        mse = np.mean((action - true_returns) ** 2)
        total_mse.append(mse)

        if verbose:
            print(f"[{pos}] MSE: {mse:.6f} | Pred: {np.round(action, 4)} | Real: {np.round(true_returns, 4)}")

    avg_error = np.mean(total_mse)
    return avg_error, total_mse
