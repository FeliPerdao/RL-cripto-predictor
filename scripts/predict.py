import numpy as np
from stable_baselines3 import PPO
from env.candle_env import CandlePredictionEnv
from stable_baselines3.common.env_util import make_vec_env


def predict(data, model_path, steps=1, window_size=10, return_only=False):
    data = data.copy().reset_index(drop=True)

    if "return" not in data.columns or "volume" not in data.columns:
        raise ValueError("❌ El dataset debe tener columnas 'return' y 'volume'")

    env = CandlePredictionEnv(data, target_step=steps, window_size=window_size)
    vec_env = make_vec_env(lambda: env, n_envs=1)
    model = PPO.load(model_path)

    env.position = len(data) - window_size
    obs = env._get_obs()
    action, _ = model.predict(obs, deterministic=True)

    if return_only:
        return action

    # Action is a single value array
    pred_return = action[0]
    direction = "⬆️" if pred_return > 0.001 else "⬇️" if pred_return < -0.001 else "➡️"

    print(f"\n🔮 Predicción vela +{steps} (variación % estimada): {pred_return:.4%} {direction}")