from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.candle_env import CandlePredictionEnv


def train_agent(data, model_path, predict_steps=3):
    # Preprocessing: prepare 'return' and 'volume' columns to be sent agent
    data = data.copy()

    #data["return"] = data["close"].pct_change().fillna(0) #% variation related to anterior candle
    #data["volume"] = data["volume"].fillna(0) #volume as is
    # Suponemos que "return" y "volume" ya vienen calculados desde main.py
    if "return" not in data.columns or "volume" not in data.columns:
        raise ValueError("❌ El dataset debe contener columnas 'return' y 'volume'")
    
    # 80/20 - rain/Test division (First 80% train - Last 20% backtest)
    train_df = data.iloc[:int(len(data) * 0.8)].copy()
    test_df = data.iloc[int(len(data) * 0.8):].copy()

    # Entorno y vectorización
    env = CandlePredictionEnv(train_df, predict_steps=predict_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Configuración y entrenamiento del modelo
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

    # Devolver datos de test
    return test_df
