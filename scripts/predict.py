import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env

# Se puede volver al anterior commit... no deve haber cambios
def predict(data, model_path, steps=3, return_only=False):
    # === Preprocesamiento ===
    data = data.copy().reset_index(drop=True)
    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)

    # Guardar precios de cierre recientes para imprimir después
    last_closes = data["close"].iloc[-10:].values
    last_price = last_closes[-1]

    # === Preparar entorno y modelo ===
    env = CandlePredictionEnv(data, predict_steps=steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO.load(model_path)

    # Posición justo antes de la predicción
    env.position = len(data) - 10
    obs = env._get_obs()
    action, _ = model.predict(obs, deterministic=True)  # acción = retornos predichos

    if return_only:
        return action

    # === Mostrar información ===
    print("\n📈 Últimos 3 precios de cierre reales:")
    for i, price in enumerate(last_closes[-3:]):
        print(f"   Vela -{3 - i}: ${price:.8f}")

    print("\n🔮 Predicción próximas velas:")
    for i, ret in enumerate(action):
        predicted_price = last_price * (1 + ret)
        direction = "⬆️" if ret > 0.001 else "⬇️" if ret < -0.001 else "➡️"
        print(f"   Vela +{i+1}: Estimado = ${predicted_price:.8f} ({ret:.4%}) {direction}")
        last_price = predicted_price
