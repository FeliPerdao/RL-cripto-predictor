from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.candle_env import CandlePredictionEnv


def train_agent(data, model_path, predict_steps=3):
    """
    Entrena un agente PPO para predecir retornos de velas con múltiples indicadores técnicos.

    Args:
        data (pd.DataFrame): Datos históricos con columnas 'close' y 'volume'.
        model_path (str): Ruta para guardar el modelo entrenado.
        predict_steps (int): Número de pasos futuros que el agente debe predecir.

    Returns:
        pd.DataFrame: Conjunto de test para validación posterior.
    """
    data = data.copy()

    data["return"] = data["close"].pct_change().fillna(0) #% variation related to anterior candle
    data["volume"] = data["volume"].fillna(0) #volume as is

    # 80/20 - train/test division (First 80% train - Last 20% backtest)
    split_index = int(len(data) * 0.8)
    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    # Entorno y vectorización
    env = CandlePredictionEnv(train_df, predict_steps=predict_steps)
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Configuración y entrenamiento del modelo
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
    )

    model.learn(total_timesteps=100_000)
    model.save(model_path)

    # Devolver datos de test
    return test_df
