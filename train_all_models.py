# train_all_models.py
import pandas as pd
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.candle_env import CandlePredictionEnv

# === Cargar combinaciones ===
with open("feature_combinations.json") as f:
    feature_sets = json.load(f)

# === Timeframes a entrenar ===
timeframes = ["1m", "3m", "5m", "15m", "1h", "1d"]
symbol = "PEPEUSDT"
data_dir = "data/historical_data"
model_dir = "models/all_features"
checkpoint_file = os.path.join(model_dir, "checkpoint.json")
os.makedirs(model_dir, exist_ok=True)

# === Cargar checkpoint existente si hay ===
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        done = set(json.load(f))
else:
    done = set()

# === Entrenamiento ===
for tf in timeframes:
    print(f"\n🧠 Entrenando modelos para {tf}...")
    df = pd.read_csv(f"{data_dir}/{symbol}_{tf}.csv")
    df["return"] = df["close"].pct_change().fillna(0)
    df["volume"] = df["volume"].fillna(0)

    count = 0
    for idx, features in enumerate(feature_sets):
        name = "-".join(features)
        model_name = f"t{tf}_{name}.zip"
        model_path = os.path.join(model_dir, model_name)

        checkpoint_id = f"{tf}|{name}"
        if checkpoint_id in done:
            continue

        try:
            env = CandlePredictionEnv(df, predict_steps=3, selected_features=features)
            vec_env = make_vec_env(lambda: env, n_envs=1)

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
            done.add(checkpoint_id)
            count += 1
            print(f"✅ Modelo guardado: {model_name}")

            # Guardar progreso
            with open(checkpoint_file, "w") as f:
                json.dump(list(done), f, indent=2)

        except Exception as e:
            print(f"❌ Fallo modelo {name}: {e}")

    print(f"\n✅ Completado {len([d for d in done if d.startswith(tf+'|')])}/8191 combinaciones para timeframe {tf}")
