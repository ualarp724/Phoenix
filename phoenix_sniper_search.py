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
    print(f"--- 🎯 BUSCANDO CONFIGURACIÓN SNIPER (Objetivo: WR > 60%, DD < 20%) ---")
    
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    df_test = df.iloc[int(len(df)*0.8):].copy()
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model = cargar_modelo()
    
    # MATRIZ DE SCALPING / ALTA PROBABILIDAD
    # Probamos Ratios más ajustados (1:1, 1:1.5) que favorecen el Win Rate
    sl_opts = [1.0, 1.5, 2.0]     
    tp_opts = [1.0, 1.2, 1.5, 2.0] # TP más cortos = Más aciertos
    umbrales = [0.80, 0.85, 0.90]  # Confianza alta
    
    print(f"{'CONF':<5} | {'SL':<4} | {'TP':<4} | {'WIN RATE':<9} | {'MAX DD':<8} | {'PROFIT':<10} | {'OPS':<5}")
    print("-" * 75)
    
    candidatos = []
    
    for umbral in umbrales:
        for sl in sl_opts:
            for tp in tp_opts:
                cap, dd, wins, ops = test_sniper(model, scaler, df_test, sl, tp, umbral)
                
                if ops < 10: continue # Ignorar si opera muy poco
                
                wr = (wins / ops) * 100
                profit = cap - config.CAPITAL_INICIAL
                
                # SOLO MOSTRAR SI CUMPLE CRITERIOS MÍNIMOS
                if wr > 50 and dd < 0.25:
                    print(f"{umbral:<5} | {sl:<4} | {tp:<4} | {wr:<8.1f}% | {dd*100:<7.1f}% | ${profit:<9.2f} | {ops:<5}")
                    
                    if wr >= 60 and dd <= 0.20:
                        candidatos.append((profit, wr, dd, umbral, sl, tp))
    
    print("-" * 75)
    if candidatos:
        # Ordenar por Win Rate
        candidatos.sort(key=lambda x: x[1], reverse=True)
        best = candidatos[0]
        print(f"🏆 MEJOR SNIPER: Conf {best[3]} | SL {best[4]} | TP {best[5]}")
        print(f"   Win Rate: {best[1]:.1f}% | DD: {best[2]*100:.1f}% | Profit: ${best[0]:.2f}")
    else:
        print("⚠️ No se encontró una configuración perfecta >60% WR y <20% DD.")
        print("Recomendación: Usa la que tenga mayor Profit de la lista de arriba.")

if __name__ == "__main__":
    ejecutar_busqueda_sniper()