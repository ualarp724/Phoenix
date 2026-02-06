import torch
import pandas as pd
import numpy as np
import joblib
import os
import phoenix_config as config
from phoenix_brain import PhoenixLSTM
from phoenix_processor import PhoenixDataProcessor

# AJUSTE PARA M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def cargar_modelo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    except:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
    return model.to(device)

def test_ratio(model, scaler, df_test, sl_mult, tp_mult):
    capital = config.CAPITAL_INICIAL
    pico = capital
    max_dd = 0
    wins, total_ops = 0, 0
    
    # Pre-cálculo
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    data_scaled = scaler.transform(df_test[features])
    device = next(model.parameters()).device
    
    # UMBRAL FIJO GANADOR
    UMBRAL = 0.85 
    
    i = config.LOOKBACK_WINDOW
    while i < len(df_test) - 50:
        window = data_scaled[i-config.LOOKBACK_WINDOW : i]
        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
        if pred.item() != 0 and conf.item() > UMBRAL and df_test['ATR'].iloc[i] > 0.15:
            # Lógica de Trade
            entry = df_test['Close'].iloc[i]
            atr = df_test['ATR'].iloc[i]
            
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
            
            # Gestión Riesgo (Lotes)
            riesgo = capital * config.RIESGO_POR_OPERACION
            lotes = max(riesgo / (sl_dist * config.VALOR_PUNTO), 0.01)
            
            # Dirección
            is_buy = (pred.item() == 1)
            sl = entry - sl_dist if is_buy else entry + sl_dist
            tp = entry + tp_dist if is_buy else entry - tp_dist
            
            # Simulación
            pnl = 0
            outcome = 0 # 0:Time, 1:Win, -1:Loss
            
            for j in range(1, 48): # 4h max
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
            if outcome == 1:
                pnl = abs(tp - entry) * lotes * config.VALOR_PUNTO
                wins += 1
            elif outcome == -1:
                pnl = -abs(entry - sl) * lotes * config.VALOR_PUNTO
            else:
                exit_price = df_test['Close'].iloc[i+j]
                pnl = (exit_price - entry) * lotes if is_buy else (entry - exit_price) * lotes
                
            capital += pnl
            total_ops += 1
            
            # DD
            if capital > pico: pico = capital
            dd = (pico - capital) / pico
            if dd > max_dd: max_dd = dd
            
            i += j
        i += 1
        
    return capital, max_dd, wins, total_ops

def ejecutar_lab_ratios():
    print(f"--- 🔬 LABORATORIO DE RATIOS (Umbral Fijo 0.85) ---")
    
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    df_test = df.iloc[int(len(df)*0.8):].copy()
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model = cargar_modelo()
    
    # MATRIZ DE PRUEBAS
    sl_opts = [1.0, 1.5, 2.0]        # Stops más ajustados o más holgados
    tp_opts = [1.5, 2.0, 3.0, 4.0]   # Targets conservadores o ambiciosos
    
    print(f"{'SL (xATR)':<10} | {'TP (xATR)':<10} | {'CAPITAL':<12} | {'MAX DD':<10} | {'OPS':<5}")
    print("-" * 60)
    
    best_score = 0
    best_cfg = ""
    
    for sl in sl_opts:
        for tp in tp_opts:
            # Filtro: El TP debe ser al menos igual al SL (Ratio 1:1 minimo)
            if tp < sl: continue 
            
            cap, dd, wins, ops = test_ratio(model, scaler, df_test, sl, tp)
            
            # Score: Buscamos Ganancia pero penalizamos DD fuerte
            # (Ganancia / DD) es una especie de Calmar Ratio simplificado
            ganancia = cap - config.CAPITAL_INICIAL
            score = ganancia / (dd + 0.01) 
            
            print(f"{sl:<10} | {tp:<10} | ${cap:<11.2f} | {dd*100:.1f}%     | {ops:<5}")
            
            if score > best_score:
                best_score = score
                best_cfg = f"SL {sl} / TP {tp}"
                
    print("-" * 60)
    print(f"🥇 MEJOR BALANCE RIESGO/BENEFICIO: {best_cfg}")

if __name__ == "__main__":
    ejecutar_lab_ratios()