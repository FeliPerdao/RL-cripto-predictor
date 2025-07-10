import gymnasium as gym
import numpy as np

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, predict_steps=3):
        super(CandlePredictionEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.predict_steps = predict_steps
        self.position = 10

        # Observación: últimas 10 velas
        # Determinar dinámicamente cuántas features hay por paso
        self.feature_columns = list(data.columns)
        self.window_size = 10
        self.obs_len = len(self.feature_columns) * self.window_size

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_len,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.predict_steps,), dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 10
        obs = self._get_obs()
        return obs, {} 


    def _get_obs(self):
        recent_data = self.data.iloc[self.position - self.window_size:self.position]
        obs_parts = []

        for col in self.feature_columns:
            series = recent_data[col].values.astype("float32")

            # Normalizar volumen
            if col == "volume":
                series = (series - series.mean()) / (series.std() + 1e-6)

            # No tocar ema_trend_up ni otras binarias
            obs_parts.append(series)

        obs = np.concatenate(obs_parts).astype("float32")
        return obs

    def step(self, action):
        future_returns = self.data.iloc[self.position:self.position + self.predict_steps]["return"].values
        action = np.clip(action, -1, 1)

        # Error cuadrático inverso como recompensa
        mse = np.mean((action - future_returns) ** 2)
        reward = -mse

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        terminated = done
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}
