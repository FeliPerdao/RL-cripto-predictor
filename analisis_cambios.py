""" import pandas as pd
import itertools

# Cargar el archivo CSV
df = pd.read_csv("results/patrones_clasificados.csv")

# Nos quedamos solo con las columnas que nos interesan
df_filtered = df.drop(columns=["timestamp", "close"])

# Listado de patrones de modelo por timeframe
pattern_cols = [col for col in df_filtered.columns if col.startswith("pattern_")]

# Listado de steps de predicción por timeframe
step_cols = [col for col in df_filtered.columns if "_step" in col]

# Generar todas las combinaciones posibles
combinations = []
for pat_cols in itertools.combinations(pattern_cols, r=3):
    for step_group in itertools.combinations(step_cols, r=3):
        if len(set(col.split("_")[0] for col in step_group)) == 1:  # mismo timeframe
            combinations.append((list(pat_cols), list(step_group)))

# Evaluar correlaciones y relaciones para cada combinación
for pat_cols, step_group in combinations:
    tf = step_group[0].split("_")[0]
    pattern_tf = f"pattern_{tf}"

    print(f"\n🔎 Timeframe: {tf.upper()} - Patrones: {', '.join(pat_cols)} - Steps: {', '.join(step_group)}")

    # Asegurarse de que pattern_{tf} esté disponible
    if pattern_tf not in df_filtered.columns:
        print(f"⚠️  {pattern_tf} no disponible. Saltando...")
        continue

    cols_to_use = ["patron"] + pat_cols + step_group
    if pattern_tf not in pat_cols:
        cols_to_use.append(pattern_tf)

    subset = df_filtered[cols_to_use].dropna()

    # Conteo de combinaciones patrón declarado vs patrón modelo
    crosstab = pd.crosstab(subset["patron"], subset[pattern_tf])
    print("\n📊 Frecuencia de patrón declarado vs patrón modelo:")
    print(crosstab)

    # Promedios de cada step por tipo de patrón declarado
    step_avg = subset.groupby("patron")[step_group].mean()
    print("\n📈 Promedio de steps por patrón declarado:")
    print(step_avg.round(4)) """
    
import pandas as pd
import itertools
import numpy as np

# Cargar CSV
df = pd.read_csv("results/patrones_clasificados.csv")
df_filtered = df.drop(columns=["timestamp", "close"])

# Detectar columnas
pattern_cols = [col for col in df_filtered.columns if col.startswith("pattern_")]
step_cols = [col for col in df_filtered.columns if "_step" in col]

# Generar combinaciones: mismo timeframe para los steps
combinations = []
for pat_cols in itertools.combinations(pattern_cols, r=3):
    for step_group in itertools.combinations(step_cols, r=3):
        tf = step_group[0].split("_")[0]
        if all(s.startswith(tf) for s in step_group):
            combinations.append((list(pat_cols), list(step_group)))

# Evaluar cada combinación
best_score = -np.inf
best_combo = None
best_step_avg = None
best_crosstab = None

for pat_cols, step_group in combinations:
    tf = step_group[0].split("_")[0]
    pattern_tf = f"pattern_{tf}"

    if pattern_tf not in df_filtered.columns:
        continue

    cols_to_use = ["patron"] + list(set(pat_cols + [pattern_tf])) + step_group
    subset = df_filtered[cols_to_use].dropna()

    if subset.empty:
        continue

    # Promedios por patrón
    step_avg = subset.groupby("patron")[step_group].mean()

    # Calcular score: varianza total entre clases para esos steps
    score = step_avg.var().sum()

    if score > best_score:
        best_score = score
        best_combo = (tf, pat_cols, step_group)
        best_step_avg = step_avg
        best_crosstab = pd.crosstab(subset["patron"], subset[pattern_tf])

# Mostrar resultados
if best_combo:
    tf, pat_cols, step_group = best_combo
    print(f"\n🏆 Mejor combinación encontrada:")
    print(f"Timeframe: {tf.upper()}")
    print(f"Patrones de modelo: {', '.join(pat_cols)}")
    print(f"Steps: {', '.join(step_group)}")
    print(f"Score total de varianza: {best_score:.4f}")

    print("\n📊 Frecuencia de patrón declarado vs patrón modelo:")
    print(best_crosstab)

    print("\n📈 Promedio de steps por patrón declarado:")
    print(best_step_avg.round(4))
else:
    print("❌ No se encontró ninguna combinación válida.")
