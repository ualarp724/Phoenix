import torch
import pandas as pd
import numpy as np
import joblib
import os
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

# --- CORRECCIÓN DE SEGURIDAD PARA MAC M4 ---
# Evita errores de memoria si el modelo se guardó con MPS
def cargar_modelo_seguro(path, input_size, hidden_layers, num_classes):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PhoenixLSTM(input_size=input_size, hidden_layers=hidden_layers, num_classes=num_classes)
    
    # Truco: Cargar en CPU primero para limpiar referencias y luego mover
    try:
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ Alerta de carga: {e}")
    
    return model.to(device)

def calcular_lotes_dinamicos(capital, precio_entrada, sl_precio):
    riesgo_dinero = capital * config.RIESGO_POR_OPERACION 
    distancia_puntos = abs(precio_entrada - sl_precio)
    if distancia_puntos == 0: return 0.01
    lotes = riesgo_dinero / (distancia_puntos * config.VALOR_PUNTO)
    return max(round(lotes, 2), 0.01)

def ejecutar_auditoria():
    print(f"--- [AUDITORÍA] Iniciando Backtest Institucional (Capital: ${config.CAPITAL_INICIAL}) ---")
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ Error: No existe el modelo. Ejecuta phoenix_brain.py primero.")
        return

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    # Usamos los datos "Test" (el último 20% que el modelo NO vio)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    
    # 8 features exactos
    model = cargar_modelo_seguro(config.MODEL_SAVE_PATH, 8, config.HIDDEN_LAYERS, 3)
    model.eval()

    capital = config.CAPITAL_INICIAL
    wins = 0
    losses = 0
    trades_log = []
    
    features_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    data_scaled = scaler.transform(df_test[features_cols])
    
    print(f"--- [SIMULACIÓN] Analizando {len(df_test)} velas desconocidas ---")

    i = config.LOOKBACK_WINDOW
    # Margen de seguridad para no salirnos del array
    while i < len(df_test) - 50: 
        
        # 1. Preparar Ventana
        # FIX DE DIMENSIONES: Asegurar que el tensor tenga la forma (1, 60, 8)
        window = data_scaled[i-config.LOOKBACK_WINDOW : i]
        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        
        # 2. Consultar al Oráculo
        with torch.no_grad():
            out = model(tensor_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            confianza, prediccion = torch.max(probs, dim=1)
            pred = prediccion.item()
            conf = confianza.item()

        # 3. Filtros
        atr_actual = df_test['ATR'].iloc[i]
        
        # Umbral de confianza
        if pred != 0 and conf > 0.60 and atr_actual > 0.15:
            
            precio_entry = df_test['Close'].iloc[i]
            sl_dist = atr_actual * config.ATR_SL_MULTIPLIER
            tp_dist = sl_dist * config.ATR_TP_MULTIPLIER 
            
            if pred == 1: # BUY
                sl = precio_entry - sl_dist
                tp = precio_entry + tp_dist
                tipo = 'BUY'
            else: # SELL
                sl = precio_entry + sl_dist
                tp = precio_entry - tp_dist
                tipo = 'SELL'
            
            lotes = calcular_lotes_dinamicos(capital, precio_entry, sl)
            
            outcome = None 
            pnl = 0
            j = 0
            
            # Recorrido del precio (4 horas max)
            for j in range(1, 48): 
                # PROTECCIÓN DE ÍNDICE FUERA DE RANGO
                current_idx = i + j
                if current_idx >= len(df_test): 
                    outcome = 'END_OF_DATA'
                    break
                
                high_fut = df_test['High'].iloc[current_idx]
                low_fut = df_test['Low'].iloc[current_idx]
                
                if tipo == 'BUY':
                    if low_fut <= sl: outcome = 'SL'; break
                    if high_fut >= tp: outcome = 'TP'; break
                else: # SELL
                    if high_fut >= sl: outcome = 'SL'; break
                    if low_fut <= tp: outcome = 'TP'; break
            
            # Resolución
            idx_cierre = min(i + j, len(df_test) - 1) # Asegurar índice válido
            
            if outcome == 'TP':
                pnl = (abs(tp - precio_entry) * lotes * config.VALOR_PUNTO)
                capital += pnl
                wins += 1
            elif outcome == 'SL':
                pnl = -(abs(precio_entry - sl) * lotes * config.VALOR_PUNTO)
                capital += pnl 
                losses += 1
            else:
                # Salida por tiempo o fin de datos
                close_final = df_test['Close'].iloc[idx_cierre]
                if tipo == 'BUY': pnl = (close_final - precio_entry) * lotes
                else: pnl = (precio_entry - close_final) * lotes
                capital += pnl
            
            trades_log.append({'Type': tipo, 'Result': outcome, 'PnL': pnl, 'Cap': capital})
            
            i += j # Saltar velas operadas
        
        i += 1

    # --- REPORTE ---
    total_trades = wins + losses
    print("\n" + "="*40)
    print(f" RESULTADO AUDITORÍA (IA Overfitted?)")
    print("="*40)
    print(f"Capital Final:   ${capital:.2f}")
    print(f"Rendimiento:     {((capital - config.CAPITAL_INICIAL)/config.CAPITAL_INICIAL)*100:.2f}%")
    print(f"Total Trades:    {len(trades_log)}")
    if len(trades_log) > 0:
        print(f"Win Rate:        {(wins/len(trades_log))*100:.1f}%")
        print(f"Trades Ganados:  {wins} | Perdidos: {losses}")
    print("="*40)

if __name__ == "__main__":
    ejecutar_auditoria()