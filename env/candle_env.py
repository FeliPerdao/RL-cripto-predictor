import gymnasium as gym
import numpy as np

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, target_step=1, window_size=10): # Hardcoded in 1 for unique candle prediction
        super(CandlePredictionEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.target_step = target_step - 1  # 0-indexed 
        self.window_size = window_size
        self.position = self.window_size

        # Observación: últimas "window_size" velas
        # Determinar dinámicamente cuántas features hay por paso
        self.feature_columns = list(data.columns)
        self.obs_len = len(self.feature_columns) * self.window_size

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_len,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32  # Hardcoded in 1 unique candle prediction
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.window_size
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
        future_returns = self.data.iloc[
            self.position:self.position + self.target_step + 1
        ]["return"].values
        
        action = np.clip(action, -1, 1) # Makes sure there is no outlier with extreme value returned

        # Rewards policy
        if len(future_returns) <= self.target_step:
            reward = -999  # Penalty if there isn't the step after the moment in training
        else:
            real = future_returns[self.target_step]
            predicted = action[0]
            reward = -((predicted - real) ** 2)

        self.position += 1
        terminated = self.position + self.target_step >= len(self.data)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

