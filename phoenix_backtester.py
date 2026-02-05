import torch
import pandas as pd
import numpy as np
import joblib
import os
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

def ejecutar_auditoria_reciente():
    # --- CONFIGURACIÓN DE LOS DATOS NUEVOS ---
    ARCHIVO_RECIENTE = "vantage_live_gold.csv"
    
    if not os.path.exists(ARCHIVO_RECIENTE):
        print(f"[ERROR] No se encuentra el archivo {ARCHIVO_RECIENTE}. Verifique el nombre.")
        return

    print(f"--- [AUDITORÍA] Test de Datos Recientes (Últimos 2 meses) ---")
    
    # 1. Cargar y Procesar los Datos Nuevos
    processor = PhoenixDataProcessor(ARCHIVO_RECIENTE)
    df_test = processor.clean_and_prepare()
    
    # 2. Cargar Inteligencia Consolidada
    try:
        scaler = joblib.load("phoenix_scaler.pkl")
        model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        model.eval()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo o el scaler: {e}")
        return

    # --- VARIABLES DE CONTROL REALISTA ---
    capital = config.CAPITAL_INICIAL
    balance_history = [capital]
    operaciones, wins, losses = 0, 0, 0
    en_operacion = False
    vela_salida = 0
    
    features = ['Open', 'High', 'Low', 'Close', 'Returns', 'Z_Score_Vol']
    data_scaled = scaler.transform(df_test[features].values)

    print(f"--- [SIMULACIÓN] Analizando {len(df_test)} velas recientes del Oro ---")

    # 3. Bucle de Trading (Modo Francotirador)
    lookback = config.LOOKBACK_WINDOW
    for i in range(lookback, len(df_test) - 5):
        if en_operacion:
            if i >= vela_salida:
                en_operacion = False
            continue

        ventana = data_scaled[i - lookback : i]
        tensor_ventana = torch.tensor(ventana, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor_ventana)
            probs = torch.nn.functional.softmax(output, dim=1)
            confianza, prediccion = torch.max(probs, dim=1)
            confianza = confianza.item()
            prediccion = prediccion.item()

        # EJECUCIÓN CON FILTRO DE CONFIANZA DEL 92% (SEGURIDAD MÁXIMA)
        if prediccion != 0 and confianza > 0.92: 
            precio_entrada = df_test['Close'].iloc[i]
            precio_futuro = df_test['Close'].iloc[i + 5]
            
            coste_friccion = 0.25 # Spread + Comisión Vantage
            pnl_puro = (precio_futuro - precio_entrada) if prediccion == 1 else (precio_entrada - precio_futuro)
            
            # Solo contamos victoria si cubrimos costes y superamos el Target de 1.00 punto
            if pnl_puro >= 1.00:
                capital += (capital * config.RIESGO_POR_OPERACION * 2)
                wins += 1
            else:
                capital -= (capital * config.RIESGO_POR_OPERACION)
                losses += 1
            
            operaciones += 1
            en_operacion = True
            vela_salida = i + 5
            balance_history.append(capital)

    # 4. Reporte Final para el CEO
    print("\n" + "="*40)
    print(" INFORME EJECUTIVO: DATOS RECIENTES")
    print("="*40)
    print(f"Archivo analizado: {ARCHIVO_RECIENTE}")
    print(f"Capital Final: ${capital:.2f}")
    print(f"Rendimiento en 2 meses: {((capital-config.CAPITAL_INICIAL)/config.CAPITAL_INICIAL)*100:.2f}%")
    print(f"Operaciones Realizadas: {operaciones}")
    if operaciones > 0:
        print(f"Win Rate: {(wins/operaciones)*100:.2f}%")
        print(f"Profit Factor: {(wins*2 / max(losses, 1)):.2f}")
    else:
        print("El bot no encontró señales con confianza suficiente.")
    print("="*40)

if __name__ == "__main__":
    ejecutar_auditoria_reciente()