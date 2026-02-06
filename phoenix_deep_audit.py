import torch
import pandas as pd
import numpy as np
import joblib
import os
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

# --- CONFIGURACIÓN A AUDITAR (LA GANADORA) ---
UMBRAL_CONFIANZA = 0.85
ATR_SL = 1.5
ATR_TP = 3.0
CAPITAL_INICIAL = 200.0
RIESGO_PCT = 0.02  # 2% Riesgo

# AJUSTE M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def cargar_modelo():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    except:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location="cpu"))
    return model.to(device)

def ejecutar_auditoria_profunda():
    print(f"--- 🕵️ AUDITORÍA FORENSE DE ESTRATEGIA ---")
    print(f"CONF: {UMBRAL_CONFIANZA} | SL: {ATR_SL}xATR | TP: {ATR_TP}xATR | RIESGO: {RIESGO_PCT*100}%")
    
    # Cargar Datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    # Usamos SOLO el Test Set (Datos que la IA nunca vio)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model = cargar_modelo()
    device = next(model.parameters()).device
    
    # Pre-cálculo features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    data_scaled = scaler.transform(df_test[features])
    
    capital = CAPITAL_INICIAL
    balance_history = [capital]
    trades = []
    
    i = config.LOOKBACK_WINDOW
    print(f"--- Analizando {len(df_test)} velas en busca de grietas... ---")
    
    while i < len(df_test) - 50:
        window = data_scaled[i-config.LOOKBACK_WINDOW : i]
        tensor_x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_x)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
        pred_idx = pred.item()
        conf_val = conf.item()
        atr = df_test['ATR'].iloc[i]
        
        # GATILLO
        if pred_idx != 0 and conf_val > UMBRAL_CONFIANZA and atr > 0.15:
            entry_price = df_test['Close'].iloc[i]
            entry_time = df_test.index[i]
            
            sl_dist = atr * ATR_SL
            tp_dist = atr * ATR_TP
            
            # Gestión Monetaria
            riesgo_usd = capital * RIESGO_PCT
            lotes = max(riesgo_usd / (sl_dist * config.VALOR_PUNTO), 0.01)
            
            is_buy = (pred_idx == 1)
            sl = entry_price - sl_dist if is_buy else entry_price + sl_dist
            tp = entry_price + tp_dist if is_buy else entry_price - tp_dist
            
            # Simulación Vela a Vela
            outcome = "TIME"
            pnl = 0
            duration = 0
            
            for j in range(1, 60): # Dejar correr hasta 5 horas (60 velas M5)
                idx = i + j
                if idx >= len(df_test): break
                
                high = df_test['High'].iloc[idx]
                low = df_test['Low'].iloc[idx]
                
                if is_buy:
                    if low <= sl: outcome = "SL"; break
                    if high >= tp: outcome = "TP"; break
                else:
                    if high >= sl: outcome = "SL"; break
                    if low <= tp: outcome = "TP"; break
                duration = j
            
            # Calcular PnL Real
            if outcome == "TP":
                pnl = abs(tp - entry_price) * lotes * config.VALOR_PUNTO
            elif outcome == "SL":
                pnl = -abs(entry_price - sl) * lotes * config.VALOR_PUNTO
            else:
                exit_price = df_test['Close'].iloc[i+duration]
                pnl = (exit_price - entry_price) * lotes if is_buy else (entry_price - exit_price) * lotes
            
            capital += pnl
            balance_history.append(capital)
            
            trades.append({
                'Time': entry_time,
                'Type': 'BUY' if is_buy else 'SELL',
                'Outcome': outcome,
                'PnL': pnl,
                'Lotes': lotes,
                'Duration': duration * 5 # Minutos
            })
            
            i += duration # Saltar velas del trade
        i += 1
        
    # --- INFORME FORENSE ---
    df_trades = pd.DataFrame(trades)
    
    if len(df_trades) == 0:
        print("❌ NO SE ENCONTRARON OPERACIONES CON ESTOS PARÁMETROS.")
        return

    # Métricas Avanzadas
    total_trades = len(df_trades)
    wins = df_trades[df_trades['PnL'] > 0]
    losses = df_trades[df_trades['PnL'] <= 0]
    
    gross_profit = wins['PnL'].sum()
    gross_loss = abs(losses['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    # Rachas
    df_trades['Win'] = df_trades['PnL'] > 0
    df_trades['Streak'] = df_trades['Win'].ne(df_trades['Win'].shift()).cumsum()
    streaks = df_trades.groupby('Streak')['Win'].agg(['first', 'count'])
    max_consecutive_wins = streaks[streaks['first']]['count'].max() if not streaks[streaks['first']].empty else 0
    max_consecutive_losses = streaks[~streaks['first']]['count'].max() if not streaks[~streaks['first']].empty else 0
    
    print("\n" + "="*50)
    print(f"📊 REPORTE DE ESTRATEGIA (PHOENIX v2.1)")
    print("="*50)
    print(f"Capital Inicial:    ${CAPITAL_INICIAL:.2f}")
    print(f"Capital Final:      ${capital:.2f} (Rendimiento: {((capital-CAPITAL_INICIAL)/CAPITAL_INICIAL)*100:.2f}%)")
    print("-" * 50)
    print(f"Total Operaciones:  {total_trades}")
    print(f"Win Rate:           {(len(wins)/total_trades)*100:.2f}%")
    print(f"Profit Factor:      {profit_factor:.2f} (Objetivo > 1.5)")
    print("-" * 50)
    print(f"Ganancia Promedio:  ${wins['PnL'].mean():.2f}")
    print(f"Pérdida Promedio:   ${losses['PnL'].mean():.2f}")
    print(f"Ratio Promedio:     1 : {abs(wins['PnL'].mean() / losses['PnL'].mean()):.2f}")
    print("-" * 50)
    print(f"🔥 Racha Ganadora Max:  {max_consecutive_wins} trades")
    print(f"❄️ Racha Perdedora Max: {max_consecutive_losses} trades (OJO AQUÍ)")
    print(f"⏳ Duración Media:      {df_trades['Duration'].mean():.0f} minutos")
    print("="*50)
    
    # Alerta de Seguridad
    if max_consecutive_losses > 6:
        print("⚠️ ADVERTENCIA: La racha de pérdidas es alta. Asegúrate de tener estómago para aguantar.")
    elif profit_factor < 1.2:
        print("⚠️ ADVERTENCIA: El Profit Factor es muy bajo. El riesgo de ruina es real.")
    else:
        print("✅ ESTRATEGIA ROBUSTA: Aprobada para fase de pruebas en vivo.")

if __name__ == "__main__":
    ejecutar_auditoria_profunda()