import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env

def predict(data, model_path, steps=3, return_only=False):
    data = data.copy().reset_index(drop=True)
    last_close_prices = data["close"].iloc[-10:].values

    env = CandlePredictionEnv(data, predict_steps=steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO.load(model_path)

    env.position = len(data) - 10
    obs = env._get_obs()
    action, _ = model.predict(obs, deterministic=True)

    if return_only:
        return action  # ⬅️ Esto permite usar la función silenciosamente

    # Modo consola normal
    print("📈 Últimos 3 precios de cierre reales:")
    for i, price in enumerate(last_close_prices[-3:]):
        print(f"   Vela -{3 - i}: ${price:.8f}")

    last_price = last_close_prices[-1]
    print("\n🔮 Predicción próximas 3 velas:")
    for i, ret in enumerate(action):
        predicted_price = last_price * (1 + ret)
        direction = "⬆️" if ret > 0.001 else "⬇️" if ret < -0.001 else "➡️"
        print(f"   Vela +{i+1}: Estimado = ${predicted_price:.8f} ({ret:.4%}) {direction}")
        last_price = predicted_price
