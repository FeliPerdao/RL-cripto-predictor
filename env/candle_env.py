import gymnasium as gym
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler

class CandlePredictionEnv(gym.Env):
    def __init__(self, data, predict_steps=3):
        super(CandlePredictionEnv, self).__init__()

        # Observación: 10 últimas filas de features seleccionadas
        self.feature_cols = [
            "return", "volume", "rsi", "ema_9", "ema_21",
            "macd", "macd_signal", "macd_diff",
            "boll_upper", "boll_lower",
            "d_price", "dd_price", "volume_change"
        ]
        
        self.predict_steps =  predict_steps
        self.position = 10
        
        self.data = self._add_features(data.reset_index(drop=True)).dropna().reset_index(drop=True)
        
        # Validación post-feature engineering
        if self.data[self.feature_cols].isnull().values.any():
            print("🔥 WARNING: Datos con NaNs después del feature engineering")
            print(self.data[self.data.isnull().any(axis=1)])
            raise ValueError("💣 Aún hay NaNs después de `_add_features()`")
        
        self.predict_steps = predict_steps
        self.position = max(10, self.data.shape[0] // 2)

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
        df["return"] = df["close"].pct_change()
        df["volume_change"] = df["volume"].pct_change()
        df["d_price"] = df["close"].diff()
        df["dd_price"] = df["d_price"].diff()

        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        df["ema_9"] = EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_21"] = EMAIndicator(df["close"], window=21).ema_indicator()

        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        boll = BollingerBands(df["close"])
        df["boll_upper"] = boll.bollinger_hband()
        df["boll_lower"] = boll.bollinger_lband()

        # Rellenamos todo de una vez con forward y backward fill
        df = df.ffill().bfill()
        
        # 💀 FILTRAR valores demasiado pequeños que causan inestabilidad numérica
        epsilon = 1e-8
        for col in ["close", "boll_upper", "boll_lower", "macd", "macd_signal", "macd_diff"]:
            df = df[df[col].abs() > epsilon]
        
        # Limpieza post-filtro
        df = df.dropna().reset_index(drop=True)
        
        # ⚠️ Check explícito: que no haya valores infinitos antes de escalar
        if not np.isfinite(df[self.feature_cols].values).all():
            bad = df[~np.isfinite(df[self.feature_cols]).all(axis=1)]
            print("❌ Valores infinitos encontrados antes del escalado:")
            print(bad)
            raise ValueError("🔥 Datos con inf o NaN antes del escalado")
        
        # 🔥 Normalizamos sólo las columnas de features
        scaler = StandardScaler()
        df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])
    
        assert not df.isnull().values.any(), "NaNs después del escalado"

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
        assert not np.isnan(obs).any(), f"Obs contiene NaNs en la posición {self.position}"
        
        if np.isnan(obs).any():
            print("💀 OBSERVACIÓN con NaNs detectada")
            print(recent)
            raise ValueError("🔥 Obs contiene NaNs en paso %d" % self.position)
        
        return obs

    def step(self, action):
        future_returns = self.data.iloc[self.position:self.position + self.predict_steps]["return"].values
        assert not np.isnan(future_returns).any(), f"future_returns tiene NaNs en pos {self.position}"
        action = np.clip(action, -1, 1)

        # Error cuadrático inverso como recompensa
        mse = np.mean((action - future_returns) ** 2)
        reward = -np.clip(mse, 0, 1)

        self.position += 1
        done = self.position + self.predict_steps >= len(self.data)

        return self._get_obs(), reward, done, False, {}
