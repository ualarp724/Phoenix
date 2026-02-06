import torch
import pandas as pd
import numpy as np
import joblib
import os
import time
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

# AJUSTE PARA M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def cargar_modelo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Aseguramos input_size=8 (Versión V2)
    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    except:
        # Fallback a CPU si hay problemas de mapeo
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
    return model.to(device)

def simular_escenario(model, scaler, df_test, umbral_confianza):
    capital = config.CAPITAL_INICIAL
    wins, losses = 0, 0
    drawdown_max = 0
    pico_capital = capital
    
    # Pre-cálculo de tensores para velocidad (Vectorización parcial)
    features_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    data_scaled = scaler.transform(df_test[features_cols])
    device = next(model.parameters()).device
    
    # Bucle rápido
    i = config.LOOKBACK_WINDOW
    while i < len(df_test) - 50:
        # Ventana
        window = data_scaled[i-config.LOOKBACK_WINDOW : i]
        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            confianza, prediccion = torch.max(probs, dim=1)
            pred = prediccion.item()
            conf = confianza.item()
        
        atr_actual = df_test['ATR'].iloc[i]
        
        # --- FILTRO DINÁMICO ---
        if pred != 0 and conf > umbral_confianza and atr_actual > 0.15:
            precio_entry = df_test['Close'].iloc[i]
            sl_dist = atr_actual * config.ATR_SL_MULTIPLIER
            tp_dist = sl_dist * config.ATR_TP_MULTIPLIER
            
            # Cálculo de Lotes (Riesgo constante)
            riesgo_dinero = capital * config.RIESGO_POR_OPERACION
            lotes = max(riesgo_dinero / (sl_dist * config.VALOR_PUNTO), 0.01)
            
            if pred == 1: # BUY
                sl = precio_entry - sl_dist
                tp = precio_entry + tp_dist
            else: # SELL
                sl = precio_entry + sl_dist
                tp = precio_entry - tp_dist
            
            # Resultado (Simplificado para velocidad)
            outcome = 0 # 0: Neutral, >0 Win, <0 Loss
            j = 0
            for j in range(1, 48): # 4 horas
                idx = i + j
                if idx >= len(df_test): break
                high = df_test['High'].iloc[idx]
                low = df_test['Low'].iloc[idx]
                
                if pred == 1:
                    if low <= sl: outcome = -1; break
                    if high >= tp: outcome = 1; break
                else:
                    if high >= sl: outcome = -1; break
                    if low <= tp: outcome = 1; break
            
            # Aplicar PnL
            pnl = 0
            if outcome == 1:
                pnl = (abs(tp - precio_entry) * lotes * config.VALOR_PUNTO)
                wins += 1
            elif outcome == -1:
                pnl = -(abs(precio_entry - sl) * lotes * config.VALOR_PUNTO)
                losses += 1
            else:
                # Time Exit
                close_exit = df_test['Close'].iloc[i+j]
                pnl = (close_exit - precio_entry) * lotes if pred == 1 else (precio_entry - close_exit) * lotes
            
            capital += pnl
            
            # Calcular Drawdown
            if capital > pico_capital: pico_capital = capital
            dd = (pico_capital - capital) / pico_capital
            if dd > drawdown_max: drawdown_max = dd
            
            i += j # Saltar velas
        i += 1
        
    return capital, wins, losses, drawdown_max

def ejecutar_optimizacion():
    print(f"--- 🧪 PHOENIX OPTIMIZER: Buscando el Santo Grial ---")
    
    # Preparar Datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model = cargar_modelo()
    
    # Rango de Pruebas
    umbrales = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    print(f"{'UMBRAL':<10} | {'CAPITAL FINAL':<15} | {'WIN RATE':<10} | {'TRADES':<8} | {'MAX DD':<10}")
    print("-" * 65)
    
    mejor_resultado = 0
    mejor_umbral = 0
    
    for u in umbrales:
        cap, w, l, dd = simular_escenario(model, scaler, df_test, u)
        total = w + l # Aproximado (sin contar time exits neutros para ratio rápido)
        wr = (w / total * 100) if total > 0 else 0
        
        print(f"{u:<10.2f} | ${cap:<14.2f} | {wr:<9.1f}% | {total:<8} | {dd*100:.1f}%")
        
        if cap > mejor_resultado:
            mejor_resultado = cap
            mejor_umbral = u
            
    print("-" * 65)
    print(f"🏆 MEJOR CONFIGURACIÓN: Umbral {mejor_umbral} (Ganancia: ${(mejor_resultado - config.CAPITAL_INICIAL):.2f})")

if __name__ == "__main__":
    ejecutar_optimizacion()