import os
import pandas as pd
from scripts.backtest import backtest
from scripts.evaluate_agent import evaluate_agent

# Config
model_dir = "models/all_features"
data_dir = "data/historical_data"
predict_steps = 3  # Cambiá esto si tus modelos fueron entrenados con más pasos
n_tests = 100

# Buscar todos los modelos .zip
model_files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]

# Ordenar por nombre para que sea predecible
model_files.sort()

print("📊 Iniciando análisis de modelos...\n")

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)

    # Extraer timeframe del nombre (asume formato: ppo_predictor_1m.zip)
    try:
        data_path = os.path.join(data_dir, "PEPEUSDT_1m.csv")

        print(f"\n🔍 Modelo: {model_file}")
        print(f"📁 Dataset: {data_path}")

        # Cargar datos
        df = pd.read_csv(data_path)

        # === BACKTEST ===
        mse, _ = backtest(df, model_path, steps=predict_steps, n_tests=n_tests, verbose=False)
        print(f"✅ MSE promedio: {mse:.6f}")

        # === EVALUATE VISUAL ===
        #print(f"📈 Mostrando gráfico de evaluación cualitativa...")
        #evaluate_agent(model_path, data_path, predict_steps=predict_steps)

    except Exception as e:
        print(f"❌ Error procesando {model_file}: {e}")
