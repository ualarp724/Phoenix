import phoenix_config as config

if config.MODEL_TYPE == "xgboost":
    raise SystemExit("Sniper search LSTM no aplica en modo XGBoost.")

import torch
import pandas as pd
import numpy as np
import joblib
import os
from phoenix_brain import PhoenixLSTM
from phoenix_processor import PhoenixDataProcessor

# AJUSTE PARA M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def cargar_modelo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PhoenixLSTM(input_size=config.INPUT_SIZE, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    except:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
    return model.to(device)

def test_sniper(model, scaler, df_test, sl_mult, tp_mult, umbral):
    capital = config.CAPITAL_INICIAL
    pico = capital
    max_dd = 0
    wins, total_ops = 0, 0
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    data_scaled = scaler.transform(df_test[features])
    device = next(model.parameters()).device
    
    i = config.LOOKBACK_WINDOW
    while i < len(df_test) - 50:
        window = data_scaled[i-config.LOOKBACK_WINDOW : i]
        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        # GATILLO
        if pred.item() != 0 and conf.item() > umbral and df_test['ATR'].iloc[i] > 0.15:
            entry = df_test['Close'].iloc[i]
            atr = df_test['ATR'].iloc[i]
            
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
            
            # GESTIÓN DE RIESGO: Si buscamos scalping (TP corto), podemos subir un poco el lote
            # pero mantenemos el 2% de riesgo base sobre el SL.
            riesgo = capital * 0.02 
            lotes = max(riesgo / (sl_dist * config.VALOR_PUNTO), 0.01)
            
            is_buy = (pred.item() == 1)
            sl = entry - sl_dist if is_buy else entry + sl_dist
            tp = entry + tp_dist if is_buy else entry - tp_dist
            
            # Simulación Rápida
            outcome = 0 
            j = 0
            for j in range(1, 36): # Max 3 horas (Scalping es más rápido)
                idx = i + j
                if idx >= len(df_test): break
                high = df_test['High'].iloc[idx]
                low = df_test['Low'].iloc[idx]
                
                if is_buy:
                    if low <= sl: outcome = -1; break
                    if high >= tp: outcome = 1; break
                else:
                    if high >= sl: outcome = -1; break
                    if low <= tp: outcome = 1; break
            
            # Resultado
            pnl = 0
            if outcome == 1:
                pnl = abs(tp - entry) * lotes * config.VALOR_PUNTO
                wins += 1
            elif outcome == -1:
                pnl = -abs(entry - sl) * lotes * config.VALOR_PUNTO
            else:
                # Cierre por tiempo
                exit_price = df_test['Close'].iloc[i+j]
                pnl = (exit_price - entry) * lotes if is_buy else (entry - exit_price) * lotes
                # Consideramos win si pnl > 0 aunque sea poco
                if pnl > 0: wins += 1 
                
            capital += pnl
            total_ops += 1
            
            if capital > pico: pico = capital
            dd = (pico - capital) / pico
            if dd > max_dd: max_dd = dd
            
            i += j
        i += 1
        
    return capital, max_dd, wins, total_ops

def ejecutar_busqueda_sniper():
    from phoenix_parameter_optimizer_v2 import FastParameterOptimizer

    print("⚠️  phoenix_sniper_search.py está deprecado. Usando optimizer v2.")
    optimizer = FastParameterOptimizer(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    optimizer.optimizar()

if __name__ == "__main__":
    ejecutar_busqueda_sniper()