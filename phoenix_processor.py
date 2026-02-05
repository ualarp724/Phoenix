import pandas as pd
import numpy as np
import os
from phoenix_config import SPREAD_MAXIMO

class PhoenixDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def clean_and_prepare(self):
        print(f"--- [PROCESADOR] Analizando estructura de {self.file_path} ---")
        
        # 1. Carga con detección de separador (Vantage usa tabuladores \t)
        df = pd.read_csv(self.file_path, sep='\t') 

        # 2. Mapeo de columnas estilo Vantage/MetaTrader
        # Buscamos las columnas que contienen los nombres clave ignorando los <>
        col_map = {}
        for col in df.columns:
            c_clean = col.replace('<', '').replace('>', '').upper()
            if 'DATE' in c_clean: col_map[col] = 'Date'
            elif 'TIME' in c_clean: col_map[col] = 'Time'
            elif 'OPEN' in c_clean: col_map[col] = 'Open'
            elif 'HIGH' in c_clean: col_map[col] = 'High'
            elif 'LOW' in c_clean: col_map[col] = 'Low'
            elif 'CLOSE' in c_clean: col_map[col] = 'Close'
            elif 'VOL' in c_clean or 'TICKVOL' in c_clean: col_map[col] = 'Volume'

        df = df.rename(columns=col_map)

        # 3. Creación del Datetime unificado (Date + Time)
        print("   > Combinando Date y Time...")
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.set_index('Datetime', inplace=True)
        
        # Limpieza de columnas sobrantes
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # --- FEATURE ENGINEERING (Nivel Senior) ---
        print("   > Generando Indicadores Cuánticos...")
        
        # Retornos logarítmicos
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Z-Score de Volatilidad (20 periodos)
        df['Std_20'] = df['Returns'].rolling(window=20).std()
        df['Z_Score_Vol'] = (df['Std_20'] - df['Std_20'].rolling(window=100).mean()) / \
                            df['Std_20'].rolling(window=100).std()

        # 4. Target para la IA (Ajustado por Spread Real de Vantage)
        # En Oro, un movimiento de 0.25 cubre comisión y spread sobradamente
        df['Cost_Threshold'] = 1 
        df['Target'] = 0 # Neutro
        
        # Miramos 5 velas al futuro (Scalping)
        future_move = df['Close'].shift(-5) - df['Close']
        df.loc[future_move > df['Cost_Threshold'], 'Target'] = 1 # COMPRA
        df.loc[future_move < -df['Cost_Threshold'], 'Target'] = 2 # VENTA

        df.dropna(inplace=True)
        return df

if __name__ == "__main__":
    if not os.path.exists("vantage_gold.csv"):
        print("[ERROR] No encuentro vantage_gold.csv. Asegúrate de que esté en la carpeta.")
    else:
        processor = PhoenixDataProcessor("vantage_gold.csv")
        data = processor.clean_and_prepare()
        print(f"\n[EXITO] Datos procesados correctamente.")
        print(f"Total de velas: {len(data)}")
        print(f"Distribución de Targets (IA):\n{data['Target'].value_counts()}")