import pandas as pd
import numpy as np
import phoenix_config as config

class PhoenixDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def clean_and_prepare(self, date_start=None, date_end=None, apply_train_filter=True, relax_factor: float = 0.0):
        print(f"--- [PROCESADOR MAESTRO] Seleccionando solo operaciones de Élite ---")
        
        try:
            df = pd.read_csv(self.file_path, sep='\t')
            if len(df.columns) < 2: df = pd.read_csv(self.file_path, sep=',')
        except: raise ValueError("Error leyendo CSV.")

        # Limpieza estándar
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
        df = df.loc[:, ~df.columns.duplicated()]
        
        if 'Time' in df.columns: df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        else: df['Datetime'] = pd.to_datetime(df['Date'])
        df.set_index('Datetime', inplace=True)
        start = date_start if date_start is not None else (getattr(config, "TRAIN_START_DATE", None) if apply_train_filter else None)
        end = date_end if date_end is not None else (getattr(config, "TRAIN_END_DATE", None) if apply_train_filter else None)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        # --- INGENIERÍA DE CARACTERÍSTICAS (INPUTS) ---
        # 1. Fuerza de Tendencia
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        df['Trend_Score'] = (df['Close'] - df['EMA_50']) / df['Close'] * 1000
        
        # 2. RSI Dinámico
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Volatilidad Relativa (NATR)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['NATR'] = (tr.rolling(14).mean() / df['Close']) * 100
        
        # 4. Volumen Relativo (¿Hay institucionales?)
        df['Vol_Rel'] = df['Volume'] / df['Volume'].rolling(50).mean()
        
        # 5. Posición en Bandas
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_Width'] = (std * 4) / sma
        df['BB_Pos'] = (df['Close'] - (sma - std*2)) / (std*4)
        
        # 6. Distancia EMA
        df['Dist_EMA'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

        # --- TARGET DE ALTA CALIDAD (LA ENSEÑANZA) ---
        # Solo queremos enseñar movimientos que cumplen TP 2.0 antes que SL 1.5
        # Y ADEMÁS, que tengan algo de volumen (evitar trampas nocturnas)
        relax_factor = float(relax_factor)
        relax_factor = max(0.0, min(relax_factor, 0.9))
        
        atr = tr.rolling(14).mean()
        tp_mult = 2.0 * (1.0 - relax_factor)
        sl_mult = 1.5 * (1.0 - relax_factor)
        tp_dist = atr * tp_mult
        sl_dist = atr * sl_mult
        
        lookahead = getattr(config, "TARGET_LOOKAHEAD_BARS", None)
        if not lookahead:
            tf = str(getattr(config, "TIMEFRAME", "M15")).upper()
            if tf == "M5":
                lookahead = 48
            elif tf == "H1":
                lookahead = 4
            else:
                lookahead = 16

        future_high = df['High'].rolling(lookahead).max().shift(-lookahead)
        future_low = df['Low'].rolling(lookahead).min().shift(-lookahead)
        
        df['Target'] = 0 # Por defecto: NO HACER NADA
        
        # Condiciones estrictas para COMPRA (1)
        # 1. Toca TP antes que SL
        # 2. RSI no está sobrecomprado (>75) al entrar
        tp_buy = df['Close'] + tp_dist
        sl_buy = df['Close'] - sl_dist
        
        rsi_buy = min(100.0, 75.0 + relax_factor * (100.0 - 75.0))
        valid_buy = (future_high > tp_buy) & (future_low > sl_buy) & (df['RSI'] < rsi_buy)
        df.loc[valid_buy, 'Target'] = 1
        
        # Condiciones estrictas para VENTA (2)
        # 1. Toca TP antes que SL
        # 2. RSI no está sobrevendido (<25) al entrar
        tp_sell = df['Close'] - tp_dist
        sl_sell = df['Close'] + sl_dist
        
        rsi_sell = max(0.0, 25.0 - relax_factor * 25.0)
        valid_sell = (future_low < tp_sell) & (future_high < sl_sell) & (df['RSI'] > rsi_sell)
        df.loc[valid_sell, 'Target'] = 2
        
        df.dropna(inplace=True)
        
        # Seleccionamos las columnas finales
        df = df[['RSI', 'Vol_Rel', 'Trend_Score', 'NATR', 'BB_Width', 'BB_Pos', 'Dist_EMA', 'Target', 'Close', 'High', 'Low']]
        
        print(f"   > Datos Maestros Generados. Velas Totales: {len(df)}")
        print(f"   > Oportunidades de Oro detectadas: {len(df[df['Target']!=0])} (El resto es ruido)")
        return df