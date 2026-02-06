import pandas as pd
import numpy as np
import phoenix_config as config

class PhoenixDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def clean_and_prepare(self):
        print(f"--- [PROCESADOR] Iniciando Protocolo de Datos: {self.file_path} ---")
        
        # 1. Carga Inteligente
        try:
            df = pd.read_csv(self.file_path, sep='\t')
            if len(df.columns) < 2: 
                df = pd.read_csv(self.file_path, sep=',')
        except Exception as e:
            raise ValueError(f"Error crítico leyendo archivo: {e}")

        # 2. Estandarización de Nombres
        col_map = {}
        for col in df.columns:
            c = col.upper().replace('<', '').replace('>', '')
            if 'DATE' in c: col_map[col] = 'Date'
            elif 'TIME' in c: col_map[col] = 'Time'
            elif 'OPEN' in c: col_map[col] = 'Open'
            elif 'HIGH' in c: col_map[col] = 'High'
            elif 'LOW' in c: col_map[col] = 'Low'
            elif 'CLOSE' in c: col_map[col] = 'Close'
            elif 'VOL' in c: col_map[col] = 'Volume'
        
        df = df.rename(columns=col_map)
        
        # --- FIX CRÍTICO: Eliminar columnas duplicadas (ej. doble volumen) ---
        df = df.loc[:, ~df.columns.duplicated()]
        # ---------------------------------------------------------------------
        
        # 3. Datetime Unificado
        if 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else:
            df['Datetime'] = pd.to_datetime(df['Date'])
        
        df.set_index('Datetime', inplace=True)
        
        # Asegurar tipos float
        cols_float = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols_float].astype(float)

        # --- FEATURE ENGINEERING ---
        
        # A. ATR
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=config.ATR_PERIOD).mean()

        # B. Retornos Logarítmicos
        df['Log_Ret'] = np.log(df['Close'] / prev_close)
        
        # C. Z-Score
        roll_std = df['Log_Ret'].rolling(window=20).std()
        df['Vol_Z'] = (roll_std - roll_std.rolling(window=100).mean()) / roll_std.rolling(window=100).std()
        
        # D. Distancia EMA200
        df['Dist_EMA200'] = df['Close'] - df['Close'].ewm(span=200).mean()

        df.dropna(inplace=True)
        print(f"   > Datos procesados. Velas útiles: {len(df)}")
        return df

if __name__ == "__main__":
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    print(df.tail())