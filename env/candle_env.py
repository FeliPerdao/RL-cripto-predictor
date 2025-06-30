import gymnasium as gym
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, predict_steps=3):
        super(CandlePredictionEnv, self).__init__()

        self.data = self._add_features(data.reset_index(drop=True))
        self.predict_steps = predict_steps
        self.position = 10

        # Observación: 10 últimas filas de features seleccionadas
        self.feature_cols = [
            "return", "volume", "rsi", "ema_9", "ema_21",
            "macd", "macd_signal", "macd_diff",
            "boll_upper", "boll_lower",
            "d_price", "dd_price", "volume_change"
        ]

        # Observación: últimas 10 variaciones % + volumenes normalizados
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10 * len(self.feature_cols),), dtype=np.float32
        )

        # Acción: predicción de próximas N velas (%), continua
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.predict_steps,), dtype=np.float32
        )
        
    def _add_features(self, df):
        df = df.copy()
        df["return"] = df["close"].pct_change().fillna(0)
        df["volume_change"] = df["volume"].pct_change().fillna(0)
        df["d_price"] = df["close"].diff().fillna(0)
        df["dd_price"] = df["d_price"].diff().fillna(0)

        df["rsi"] = RSIIndicator(df["close"], window=14).rsi().fillna(0)
        df["ema_9"] = EMAIndicator(df["close"], window=9).ema_indicator().fillna(method='bfill')
        df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator().fillna(method='bfill')

        macd = MACD(df["close"])
        df["macd"] = macd.macd().fillna(0)
        df["macd_signal"] = macd.macd_signal().fillna(0)
        df["macd_diff"] = macd.macd_diff().fillna(0)

        boll = BollingerBands(df["close"])
        df["boll_upper"] = boll.bollinger_hband().fillna(0)
        df["boll_lower"] = boll.bollinger_lband().fillna(0)
        
        return df

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 10
        obs = self._get_obs()
        return obs, {}


    def _get_obs(self):
        recent = self.data.iloc[self.position - 10:self.position][self.feature_cols]
        obs = recent.values.flatten().astype("float32")
        return obs

    def step(self, action):
        future_returns = self.data.iloc[self.position:self.position + self.predict_steps]["return"].values
        action = np.clip(action, -1, 1)

        # Error cuadrático inverso como recompensa
        mse = np.mean((action - future_returns) ** 2)
        reward = -mse

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        return self._get_obs(), reward, done, False, {}
