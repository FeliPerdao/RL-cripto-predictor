import gymnasium as gym
import numpy as np

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, predict_steps=3):
        super(CandlePredictionEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.predict_steps = predict_steps
        self.position = 10

        # Observación: últimas 10 variaciones % + volumenes normalizados
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )

        # Acción: predicción de próximas N velas (%), continua
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.predict_steps,), dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 10
        return self._get_obs()

    def _get_obs(self):
        recent_data = self.data.iloc[self.position - 10:self.position].copy()
        pct_returns = recent_data["return"].values.astype("float32")
        volume = recent_data["volume"].values.astype("float32")
        norm_volume = (volume - volume.mean()) / (volume.std() + 1e-6)
        obs = np.concatenate([pct_returns, norm_volume]).astype("float32")
        return obs

    def step(self, action):
        future_returns = self.data.iloc[self.position:self.position + self.predict_steps]["return"].values
        action = np.clip(action, -1, 1)

        # Error cuadrático inverso como recompensa
        mse = np.mean((action - future_returns) ** 2)
        reward = -mse

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        return self._get_obs(), reward, done, {}
