from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.candle_env import CandlePredictionEnv


def train_agent(data, model_path, predict_steps=3):
    # Preprocesamiento: asegurarse de que 'return' y 'volume' estén listos
    data = data.copy()
    print("🔍 Columnas recibidas:", data.columns.tolist())

    data["return"] = data["close"].pct_change().fillna(0)
    data["volume"] = data["volume"].fillna(0)

    # División 80/20
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
