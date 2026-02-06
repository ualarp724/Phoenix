# --- PHOENIX PROJECT: GLOBAL CONFIGURATION (PRODUCCIÓN V1) ---

# GESTIÓN DE CAPITAL
CAPITAL_INICIAL = 200.0      
RIESGO_POR_OPERACION = 0.02  # 2% Riesgo por operación (Agresivo pero controlado)

# PARÁMETROS DE MERCADO (XAUUSD)
SYMBOL = "XAUUSD"
TIMEFRAME = "M5"
VALOR_PUNTO = 1.0   # Verfica en tu broker si 1 lote estándar mueve $1 o $10 por punto

# AI MODEL PARAMS (M4 OPTIMIZED)
LOOKBACK_WINDOW = 60
HIDDEN_LAYERS = [128, 64, 32]
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 4096 

# --- REGLAS DE ORO (DESCUBIERTAS EN LABORATORIO) ---
UMBRAL_CONFIANZA = 0.85   # Solo operamos si la IA tiene 85% de certeza
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5   # Stop Loss ajustado a la volatilidad
ATR_TP_MULTIPLIER = 3.0   # Buscamos Home Runs (Ganar 3 veces lo arriesgado)

# RUTAS
DATA_RAW = "vantage_gold.csv"
MODEL_SAVE_PATH = "phoenix_brain.pth"
SCALER_SAVE_PATH = "phoenix_scaler.pkl"