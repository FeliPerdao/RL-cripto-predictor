from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.candle_env import CandlePredictionEnv

def train_agent(data, model_path, predict_steps=3):
    env = CandlePredictionEnv(data, predict_steps=predict_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )

    model.learn(total_timesteps=100_000)
    model.save(model_path)
