import torch
import pandas as pd
import numpy as np
import joblib
import time
import os
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

# --- CONFIGURACIÓN DE OPERACIÓN REAL ---
CONFIANZA_MINIMA = 0.92  # Subimos un poco más para seguridad real
ARCHIVO_MT5 = "vantage_live_gold.csv" # El archivo que MT5 debe actualizar

def cargar_activos():
    print("--- [SISTEMA] Cargando Cerebro Phoenix... ---")
    scaler = joblib.load("phoenix_scaler.pkl")
    model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    return model, scaler

def analizar_mercado_actual(model, scaler):
    try:
        # 1. Leer el archivo que viene de MT5
        processor = PhoenixDataProcessor(ARCHIVO_MT5)
        df = processor.clean_and_prepare()
        
        # 2. Tomar las últimas 60 velas (Lookback)
        features = ['Open', 'High', 'Low', 'Close', 'Returns', 'Z_Score_Vol']
        data_scaled = scaler.transform(df[features].tail(config.LOOKBACK_WINDOW).values)
        
        tensor_ventana = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
        
        # 3. Predicción de IA
        with torch.no_grad():
            output = model(tensor_ventana)
            probs = torch.nn.functional.softmax(output, dim=1)
            confianza, prediccion = torch.max(probs, dim=1)
            
        return prediccion.item(), confianza.item(), df['Close'].iloc[-1]
    except Exception as e:
        print(f"[ERROR] Esperando datos válidos de MT5... {e}")
        return 0, 0, 0

def iniciar_bot():
    model, scaler = cargar_activos()
    print("--- [PHOENIX LIVE] Monitor de Mercado Iniciado ---")
    
    ultima_vela = None
    
    while True:
        # Analizamos cada 30 segundos para no saturar la CPU
        pred, conf, precio = analizar_mercado_actual(model, scaler)
        
        if pred != 0 and conf > CONFIANZA_MINIMA:
            tipo = "COMPRA" if pred == 1 else "VENTA"
            print(f"🔥 [ALERTA] SEÑAL DETECTADA: {tipo} | Precio: {precio} | Confianza: {conf:.4f}")
            print(f"💰 [ORDEN] Abrir 0.01 lotes en Vantage. TP: 1.00 punto | SL: 0.50 puntos.")
            
            # Aquí esperaríamos 5 minutos para la siguiente vela
            time.sleep(300) 
        
        time.sleep(30)

if __name__ == "__main__":
    iniciar_bot()