# --- PHOENIX ULTIMATE CONFIG ---
import os

# GESTIÓN DE CAPITAL
CAPITAL_INICIAL = 200.0
TARGET_DAILY_USD = 4.0
EXPECTED_DAILY_USD = 1.71
RISK_MULTIPLIER = 1.0
BTC_RISK_MULTIPLIER = 1.0
RIESGO_POR_OPERACION = 0.015  # 1.5% por operación máximo
CAPITAL_PROTECCIÓN = 50.0    # Nunca bajar de esto

# MERCADO
SYMBOL = "XAUUSD"
DATA_RAW = "m5.csv"
LIVE_FILE = "vantage_live_gold.csv"
TIMEFRAME = "M5"
TARGET_LOOKAHEAD_BARS = 48
TRAIN_START_DATE = "2024-07-01"
TRAIN_END_DATE = "2025-06-30"
VAL_START_DATE = "2025-07-01"
VAL_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-02-06"

# ARQUITECTURA UNIFICADA DE FEATURES
FEATURES = [
    'RSI', 'Vol_Rel', 'Trend_Score', 'NATR', 'BB_Width', 'BB_Pos', 'Dist_EMA',
    'M15_Trend_Score', 'M15_RSI', 'M15_BB_Pos',
    'H1_Trend_Score', 'H1_RSI', 'H1_BB_Pos',
]
INPUT_SIZE = len(FEATURES)

# CEREBRO (IA)
MODEL_TYPE = "ensemble"  # "lstm" | "xgboost" | "ensemble"
LOOKBACK_WINDOW = 60
HIDDEN_LAYERS = [256, 128]
LEARNING_RATE = 0.0005
EPOCHS = 50
BATCH_SIZE = 128
DATALOADER_WORKERS = 0
PRED_BATCH_SIZE = 1024
CPU_THREADS = None
USE_TORCH_COMPILE = True
BACKTEST_IGNORE_TIME_NEWS = False
DROPOUT_RATE = 0.3
VAL_SPLIT = 0.2  # 20% validación durante entrenamiento
EARLY_STOPPING_PATIENCE = 8  # Épocas sin mejora antes de parar
FOCAL_GAMMA = 2.0
CALIBRATION_PATH = "calibration.json"

# REGLAS DE SNIPER (PARÁMETROS A OPTIMIZAR)
UMBRAL_CONFIANZA = 0.19100153716580462
UMBRAL_BUY = 0.20
UMBRAL_SELL = 0.20
CONFIDENCE_PERCENTILE = 0.20
ATR_SL_MULTIPLIER = 1.4790540669956496
ATR_TP_MULTIPLIER = 1.5654568472090737

# FILTROS AVANZADOS
USE_REGIME_FILTER = False
USE_MTF_CONFIRM = True
MTF_TIMEFRAME = "1H"
MTF_PREFIX = "H1_"
MTF_CONFIGS = [("M15_", "15min"), ("H1_", "1h")]
USE_M15_FILTER = True
M15_TREND_MIN = 0.0
M15_TREND_NEUTRAL_MAX = 0.5

REGIME_TREND_THRESHOLD = 1.0
REGIME_BB_WIDTH_THRESHOLD = 0.006

RSI_PULLBACK_MAX = 70
BB_PULLBACK_MAX = 0.45

RSI_RANGE_LOW = 35
RSI_RANGE_HIGH = 65
BB_RANGE_LOW = 0.25
BB_RANGE_HIGH = 0.75

# MTF CONFIRM (más permisivo para aumentar señales)
MTF_TREND_MIN = 0.0
MTF_ENFORCE_TREND = False
MTF_RSI_MAX_BUY = 80
MTF_RSI_MIN_SELL = 20
MTF_BB_POS_MAX_BUY = 0.8
MTF_BB_POS_MIN_SELL = 0.2

# RUTAS
XGB_MODEL_PATH = "phoenix_xgb.pkl"
LGBM_MODEL_PATH = "phoenix_lgbm.pkl"
MODEL_SAVE_PATH = XGB_MODEL_PATH
SCALER_SAVE_PATH = "phoenix_scaler.pkl"
MODEL_BEST_SAVE_PATH = "phoenix_brain_best.pth"

# CONFIG MULTIACTIVO
ASSETS = {
    "XAUUSD": {
        "data_raw": "m5.csv",
        "live_file": "vantage_live_gold.csv",
        "xgb_model": "phoenix_xgb.pkl",
        "lgbm_model": "phoenix_lgbm.pkl",
        "scaler": "phoenix_scaler.pkl",
        "train_start": "2024-07-01",
        "train_end": "2025-06-30",
        "val_start": "2025-07-01",
        "val_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-02-06",
        "ensemble_weights": {"xgboost": 0.4, "lightgbm": 0.6},
        "risk_multiplier": 1.0,
    },
    "BTCUSD": {
        "data_raw": "vantage_btc1.csv",
        "live_file": "vantage_live_btc.csv",
        "xgb_model": "phoenix_btc_xgb.pkl",
        "lgbm_model": "phoenix_btc_lgbm.pkl",
        "scaler": "phoenix_btc_scaler.pkl",
        "train_start": "2024-07-01",
        "train_end": "2025-06-30",
        "val_start": "2025-07-01",
        "val_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-02-06",
        "ensemble_weights": {"xgboost": 0.3, "lightgbm": 0.7},
        "use_mtf_confirm": False,
        "risk_multiplier": 0.33,
    },
    "EURUSD": {
        "data_raw": "vantage_eurusd.csv",
        "live_file": "vantage_live_eurusd.csv",
        "xgb_model": "phoenix_eurusd_xgb.pkl",
        "lgbm_model": "phoenix_eurusd_lgbm.pkl",
        "scaler": "phoenix_eurusd_scaler.pkl",
        "train_start": "2022-01-01",
        "train_end": "2023-12-31",
        "val_start": "2024-01-01",
        "val_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-02-06",
        "ensemble_weights": {"xgboost": 0.4, "lightgbm": 0.6},
        "use_mtf_confirm": True,
        "risk_multiplier": 1.0,
    },
    "NAS100": {
        "data_raw": "vantage_nas100.csv",
        "live_file": "vantage_live_nas100.csv",
        "xgb_model": "phoenix_nas100_xgb.pkl",
        "lgbm_model": "phoenix_nas100_lgbm.pkl",
        "scaler": "phoenix_nas100_scaler.pkl",
        "train_start": "2024-01-01",
        "train_end": "2024-12-31",
        "val_start": "2025-01-01",
        "val_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-02-06",
        "ensemble_weights": {"xgboost": 0.4, "lightgbm": 0.6},
        "use_mtf_confirm": True,
        "risk_multiplier": 1.0,
    },
}


def apply_asset(symbol: str) -> None:
    asset = ASSETS.get(symbol)
    if not asset:
        raise ValueError(f"Símbolo no configurado: {symbol}")
    global SYMBOL, DATA_RAW, LIVE_FILE, XGB_MODEL_PATH, LGBM_MODEL_PATH, MODEL_SAVE_PATH, SCALER_SAVE_PATH, ENSEMBLE_WEIGHTS
    global USE_MTF_CONFIRM, RISK_MULTIPLIER
    global TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, TEST_START_DATE, TEST_END_DATE

    SYMBOL = symbol
    DATA_RAW = asset["data_raw"]
    LIVE_FILE = asset.get("live_file", LIVE_FILE)
    XGB_MODEL_PATH = asset["xgb_model"]
    LGBM_MODEL_PATH = asset["lgbm_model"]
    MODEL_SAVE_PATH = XGB_MODEL_PATH
    SCALER_SAVE_PATH = asset["scaler"]
    TRAIN_START_DATE = asset["train_start"]
    TRAIN_END_DATE = asset["train_end"]
    VAL_START_DATE = asset["val_start"]
    VAL_END_DATE = asset["val_end"]
    TEST_START_DATE = asset["test_start"]
    TEST_END_DATE = asset["test_end"]
    ENSEMBLE_WEIGHTS = asset.get("ensemble_weights", ENSEMBLE_WEIGHTS)
    USE_MTF_CONFIRM = asset.get("use_mtf_confirm", USE_MTF_CONFIRM)
    RISK_MULTIPLIER = asset.get("risk_multiplier", RISK_MULTIPLIER)

# XGBOOST
XGB_PARAMS = {
    "n_estimators": 550,
    "max_depth": 4,
    "learning_rate": 0.07156167661312729,
    "subsample": 0.922527341976063,
    "colsample_bytree": 0.76343204904729,
    "objective": 'multi:softprob',
    "num_class": 3,
    "eval_metric": 'mlogloss',
    "random_state": 42,
    "min_child_weight": 24,
    "gamma": 0.07813986732768911,
    "reg_alpha": 0.08968991883376717,
    "reg_lambda": 1.9274850235756338,
    "max_delta_step": 0,
}

# LIGHTGBM
LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# ENSEMBLE (VOTACIÓN)
ENSEMBLE_WEIGHTS = {
    "xgboost": 0.4,
    "lightgbm": 0.6,
}
ENSEMBLE_REQUIRE_CONSENSUS = False

# FILTROS DE TRADING
MIN_ATR_THRESHOLD = 0.05  # No operar si ATR es muy bajo
HORA_INICIO = 10
HORA_CIERRE = 20
MAX_TRADES_PER_HOUR = 3   # Máx 3 operaciones/hora
USE_TIME_FILTER = False
MIN_VOL_REL = 0.8


# --- LÍMITES CFO (NO NEGOCIABLES) ---
MAX_EFFECTIVE_LEVERAGE = 50
MAX_EXPOSURE_USD = 10000

DEFAULT_LOT_SIZE = 0.01
MAX_LOT_SIZE = 0.03
MIN_LOT_SIZE = 0.01
MAX_OPEN_POSITIONS = 3
MAX_TRADES_PER_DAY = 8

MAX_RISK_PER_TRADE_PCT = 0.015
MAX_DAILY_LOSS_USD = 12.0
MAX_WEEKLY_LOSS_USD = 30.0
DAILY_PROFIT_TARGET_USD = 4.0

MIN_STOP_LOSS_PIPS = 12
MAX_STOP_LOSS_PIPS = 30
DEFAULT_STOP_LOSS_PIPS = 20
MIN_TAKE_PROFIT_PIPS = 18
RISK_REWARD_RATIO_MIN = 1.3
RISK_REWARD_RATIO_TARGET = 1.5

ENABLE_TRAILING_STOP = True
TRAILING_STOP_ACTIVATION_PIPS = 15
TRAILING_STOP_DISTANCE_PIPS = 10

ALLOWED_PAIRS = [
	'EURUSD',
	'GBPUSD',
	'USDJPY',
	'AUDUSD',
	'USDCHF',
	'XAUUSD',
    'BTCUSD',
    'NAS100',
]

TRADING_HOURS_UTC = {
	'start': 7,
	'end': 21,
}
AVOID_ASIAN_SESSION = True
AVOID_FRIDAY_AFTER_16 = True
AVOID_SUNDAY_OPEN = True

# --- NEWS FILTER ---
NEWS_EVENTS_FILE = "news_events.json"
NEWS_FEED_FILE = "news_feed.csv"
NEWS_CURRENCIES = ["USD", "XAU"]
STOP_BEFORE_HIGH_IMPACT_NEWS_MIN = 30
RESUME_AFTER_NEWS_MIN = 15

# LOGGING
VERBOSE = True
SAVE_METRICS = True
METRICS_FILE = "trading_metrics.csv"

# COMPAT LEGACY
VALOR_PUNTO = 1.0