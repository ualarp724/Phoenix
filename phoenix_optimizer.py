import phoenix_config as config

if config.MODEL_TYPE == "xgboost":
    raise SystemExit("Optimizer LSTM no aplica en modo XGBoost.")

import torch
import pandas as pd
import numpy as np
import gc
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM, preparar_secuencias
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# --- CONFIGURACIÓN DE BÚSQUEDA ---
UMBRALES_TEST = [0.50, 0.55, 0.60, 0.65, 0.70] # Niveles de exigencia a probar
CAPITAL_BASE = 200.0
LEVERAGE = 500.0
COMISION_0_01 = 0.06

# Ajustes de Entreno
DIAS_ENTRENAMIENTO = 600
BATCH_SIZE = 64
EPOCHS = 60 

if torch.backends.mps.is_available(): DEVICE = torch.device("mps"); print("🚀 M4 GPU ACTIVA")
else: DEVICE = torch.device("cpu")

def calcular_pesos(y_tensor):
    classes, counts = np.unique(y_tensor.numpy(), return_counts=True)
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum(), dtype=torch.float32).to(DEVICE)

def entrenar_modelo(df_train):
    print(f"🧠 Entrenando IA Base ({EPOCHS} épocas)...")
    try:
        X_base, y_base, scaler = preparar_secuencias(df_train)
        weights = calcular_pesos(y_base)
        loader = DataLoader(TensorDataset(X_base, y_base), batch_size=BATCH_SIZE, shuffle=True)
        
        model = PhoenixLSTM(input_size=7, hidden_layers=[128, 64], num_classes=3).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss(weight=weights)
        
        model.train()
        for ep in range(EPOCHS):
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
        model.eval()
        return model, scaler
    except Exception as e:
        print(f"❌ Error Entreno: {e}")
        return None, None

def simular_escenario(model, tensor_test, df_test, umbral):
    capital = CAPITAL_BASE
    wins, ops = 0, 0
    drawdown_max = 0.0
    peak = capital
    
    # Bucle rápido vectorizado (simulado)
    # Para precisión, usamos el bucle lento intra-vela
    for i in range(config.LOOKBACK_WINDOW, len(df_test)-20):
        if capital < 20: break # Quemada
        
        # Filtro Horario
        h = df_test.index[i].hour
        if h < 9 or h >= 19: continue

        window = tensor_test[i-config.LOOKBACK_WINDOW:i].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.nn.functional.softmax(model(window), dim=1)
            conf, pred = torch.max(prob, dim=1)
            conf, pred = conf.item(), pred.item()
        
        if pred != 0 and conf > umbral:
            # DATOS M15
            price = df_test['Close'].iloc[i]
            atr = df_test['NATR'].iloc[i] * price / 100
            
            # Targets M15
            sl_dist = max(atr * 1.5, 1.0)
            tp_dist = max(atr * 2.5, 2.0)
            
            # LOTE FIJO 0.01 (Para testear la estrategia pura, sin martingalas)
            lotes = 0.01
            
            margin = (price * 100 * lotes) / LEVERAGE
            if capital > margin:
                # Simulación Intra-Vela Pesimista
                pnl = 0
                tp_p = price + tp_dist if pred == 1 else price - tp_dist
                sl_p = price - sl_dist if pred == 1 else price + sl_dist
                
                for j in range(1, 13): # 3 horas futuro
                    hi = df_test['High'].iloc[i+j]
                    lo = df_test['Low'].iloc[i+j]
                    
                    if pred == 1:
                        if lo <= sl_p: pnl = -sl_dist*100*lotes; break
                        if hi >= tp_p: pnl = tp_dist*100*lotes; break
                    else:
                        if hi >= sl_p: pnl = -sl_dist*100*lotes; break
                        if lo <= tp_p: pnl = tp_dist*100*lotes; break
                
                # Cierre tiempo
                if pnl == 0:
                    exit_p = df_test['Close'].iloc[i+12]
                    pnl = (exit_p - price)*100*lotes if pred == 1 else (price - exit_p)*100*lotes
                
                neto = pnl - COMISION_0_01
                capital += neto
                ops += 1
                if neto > 0: wins += 1
                
                # Update Drawdown
                if capital > peak: peak = capital
                dd = (peak - capital) / peak
                if dd > drawdown_max: drawdown_max = dd

    return capital, ops, wins, drawdown_max

def ejecutar_optimizador():
    from phoenix_parameter_optimizer_v2 import FastParameterOptimizer

    print("⚠️  phoenix_optimizer.py está deprecado. Usando optimizer v2.")
    optimizer = FastParameterOptimizer(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    optimizer.optimizar()

if __name__ == "__main__":
    ejecutar_optimizador()