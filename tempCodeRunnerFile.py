for tf in TIMEFRAMES:
    file_path = f"data/historical_data/PEPEUSDT_{tf}.csv"
    if not os.path.exists(file_path):
        logging.info(f"🔽 Descargando velas {tf}...")
        download_binance_ohlcv("PEPE/USDT", tf)
    else:
        logging.info(f"✅ Archivo existente para {tf}, usando datos locales.")

# Paso 2: Calcular variaciones porcentuales y guardar los dataframes
for tf in TIMEFRAMES:
    file_path = f"data/historical_data/PEPEUSDT_{tf}.csv"
    df = pd.read_csv(file_path)
    df = df.drop(columns=["timestamp"], errors="ignore")
    df["return"] = df["close"].pct_change().fillna(0)
    df = df[["return"]]  # solo usamos el retorno
    dataframes[tf] = df
    logging.info(f"📈 Datos procesados para {tf}")

# Paso 3: Entrenar modelos de predicción continua para próximas 3 velas
for tf in TIMEFRAMES:
    logging.info(f"🧠 Entrenando modelo para {tf}...")
    model_path = f"models/ppo_predictor_{tf}"
    train_agent(dataframes[tf], model_path)
    logging.info(f"✅ Modelo entrenado para {tf}")