import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import warnings
import numpy as np
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi, _add_single_mtf
from phoenix_backtester_pro import ProfessionalBacktester
import phoenix_config as config

# CONFIGURACIÓN DE APRENDIZAJE CONTINUO
LR_FINE_TUNING = 0.00005  # 20 veces más lento que el entrenamiento normal (Cirugía de precisión)
EPOCHS_ACTIVE = 15        # Pocas épocas para no sobreajustar
BATCH_SIZE = 256          # Lotes más pequeños para generalizar mejor

# --- HARDENING: FOCAL LOSS ---
# Esta función penaliza los errores en predicciones "seguras".
# Obliga a la IA a no confiarse y buscar el 3% con certeza matemática.
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

def _build_sequences(df, scaler):
    features = config.FEATURES
    data = df[features].values
    target = df['Target'].values
    data_scaled = scaler.transform(data)
    X, y = [], []
    lookback = config.LOOKBACK_WINDOW
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i])
        y.append(target[i])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def iniciar_auto_mejora():
    if config.MODEL_TYPE == "xgboost":
        return _active_learning_xgb()
    # Detectar M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- [AUTONOMOUS LEARNING] Iniciando Protocolo de Mejora Continua | {device} ---")
    
    # 1. Cargar Cerebro Existente
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ [ERROR] No existe un cerebro base. Ejecuta phoenix_brain.py primero.")
        return

    print("   > Cargando conocimientos previos...")
    from phoenix_brain import PhoenixLSTM
    model = PhoenixLSTM(input_size=config.INPUT_SIZE, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    
    # 2. Cargar Nuevos Datos (Simulación de "Lo que pasó esta semana")
    # Usamos 'vantage_live_gold.csv' como la fuente de nuevos datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare(
        date_start=getattr(config, "VAL_START_DATE", None),
        date_end=getattr(config, "VAL_END_DATE", None),
        apply_train_filter=False,
    )

    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]

    prev_mtf = config.USE_MTF_CONFIRM
    config.USE_MTF_CONFIRM = True
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    if "M15_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "M15_", "15min")
    if "H1_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "H1_", "1h")
    config.USE_MTF_CONFIRM = prev_mtf

    for col, default in [
        ("M15_Trend_Score", 0.0), ("M15_RSI", 50.0), ("M15_BB_Pos", 0.5),
        ("H1_Trend_Score", 0.0), ("H1_RSI", 50.0), ("H1_BB_Pos", 0.5),
    ]:
        if col not in df.columns:
            df[col] = default
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]

    prev_mtf = config.USE_MTF_CONFIRM
    config.USE_MTF_CONFIRM = True
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    if "M15_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "M15_", "15min")
    if "H1_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "H1_", "1h")
    config.USE_MTF_CONFIRM = prev_mtf

    for col, default in [
        ("M15_Trend_Score", 0.0), ("M15_RSI", 50.0), ("M15_BB_Pos", 0.5),
        ("H1_Trend_Score", 0.0), ("H1_RSI", 50.0), ("H1_BB_Pos", 0.5),
    ]:
        if col not in df.columns:
            df[col] = default
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]
    prev_mtf = config.USE_MTF_CONFIRM
    config.USE_MTF_CONFIRM = True
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    if "M15_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "M15_", "15min")
    if "H1_Trend_Score" not in df.columns:
        df = _add_single_mtf(df, "H1_", "1h")
    config.USE_MTF_CONFIRM = prev_mtf

    for col, default in [
        ("M15_Trend_Score", 0.0), ("M15_RSI", 50.0), ("M15_BB_Pos", 0.5),
        ("H1_Trend_Score", 0.0), ("H1_RSI", 50.0), ("H1_BB_Pos", 0.5),
    ]:
        if col not in df.columns:
            df[col] = default
    
    # IMPORTANTE: Usamos el Scaler original para que la IA entienda los datos igual que antes
    # (No re-entrenamos el scaler, solo transformamos)
    # Nota: Aquí reutilizamos la lógica de brain pero asumiendo que el scaler existe
    X_train, y_train = _build_sequences(df, scaler)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Configurar la "Cirugía" (Optimizador Lento + Focal Loss)
    optimizer = optim.Adam(model.parameters(), lr=LR_FINE_TUNING)
    criterion = FocalLoss(gamma=2.5) # Gamma alto = Exigencia alta

    print(f"--- [HARDENING] Refinando estrategia con {len(X_train)} nuevas velas ---")
    
    model.train()
    for epoch in range(EPOCHS_ACTIVE):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Mejora [{epoch+1}/{EPOCHS_ACTIVE}] - Loss (Focal): {total_loss/len(train_loader):.6f}")

    # 4. Guardar la versión mejorada (Sobrescribe la anterior)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"--- [EXITO] Conocimiento integrado. El bot es ahora más inteligente. ---")


def _active_learning_xgb():
    try:
        import xgboost as xgb
    except Exception as exc:
        print(f"❌ XGBoost no disponible: {exc}")
        return

    print("--- [AUTONOMOUS LEARNING] XGBoost fine-tuning en VAL ---")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ [ERROR] No existe el modelo XGBoost base.")
        return

    model = joblib.load(config.MODEL_SAVE_PATH)
    backup_path = f"{config.MODEL_SAVE_PATH}.bak"
    joblib.dump(model, backup_path)
    scaler = joblib.load(config.SCALER_SAVE_PATH)

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare(
        date_start=getattr(config, "VAL_START_DATE", None),
        date_end=getattr(config, "VAL_END_DATE", None),
        apply_train_filter=False,
    )

    features = config.FEATURES
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    if getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    data = df[features].values
    target = df["Target"].values
    data_scaled = scaler.transform(data)
    lookback = config.LOOKBACK_WINDOW
    X_val, y_val = [], []
    for i in range(lookback, len(data_scaled)):
        X_val.append(data_scaled[i - lookback:i].reshape(-1))
        y_val.append(target[i])
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    if len(X_val) == 0:
        print("⚠️ Sin datos para active learning.")
        return

    params = dict(config.XGB_PARAMS)
    params["n_estimators"] = 50
    params["learning_rate"] = min(params.get("learning_rate", 0.05), 0.03)
    model_update = xgb.XGBClassifier(**params)
    model_update.fit(X_val, y_val, xgb_model=model.get_booster(), verbose=False)

    joblib.dump(model_update, config.MODEL_SAVE_PATH)
    # Evaluar si mejora en VAL
    try:
        processor = PhoenixDataProcessor(config.DATA_RAW)
        df_eval = processor.clean_and_prepare(
            date_start=getattr(config, "VAL_START_DATE", None),
            date_end=getattr(config, "VAL_END_DATE", None),
            apply_train_filter=False,
        )
        df_eval = add_mtf_features_multi(df_eval, config.MTF_CONFIGS)
        for col in config.FEATURES:
            if col not in df_eval.columns:
                df_eval[col] = 0.0
        df_eval = df_eval.dropna(subset=config.FEATURES + ["Target"]).copy()
        backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
        result_post = backtester.backtest_period(df_eval)
        profit_post = result_post["profit"] if result_post else float("-inf")

        joblib.dump(joblib.load(backup_path), config.MODEL_SAVE_PATH)
        backtester_pre = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
        result_pre = backtester_pre.backtest_period(df_eval)
        profit_pre = result_pre["profit"] if result_pre else float("-inf")

        if profit_post >= profit_pre:
            joblib.dump(model_update, config.MODEL_SAVE_PATH)
            print("✅ Active learning XGBoost mejoró o igualó. Se mantiene.")
        else:
            print("⚠️ Active learning XGBoost empeoró. Revertido a backup.")
    except Exception as exc:
        print(f"⚠️ Evaluación post-active learning falló: {exc}")
        joblib.dump(joblib.load(backup_path), config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    iniciar_auto_mejora()