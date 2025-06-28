import gymnasium as gym
import numpy as np

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, predict_steps=3):
        super(CandlePredictionEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.predict_steps = predict_steps
        self.position = 10

        # Observación: últimas 10 variaciones %
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
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
        obs = self.data.iloc[self.position - 10:self.position]["return"].values.astype("float32")
        return obs

    def step(self, action):
        future_returns = self.data.iloc[self.position:self.position + self.predict_steps]["return"].values
        action = np.clip(action, -1, 1)

        # Error cuadrático inverso como recompensa
        mse = np.mean((action - future_returns) ** 2)
        reward = -mse  # cuanto menor el error, mayor la recompensa

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        return self._get_obs(), reward, done, {}
