import pandas as pd
import numpy as np
import phoenix_config as config

def auditar_ratios():
    print(f"--- [PHOENIX AUDIT] Buscando la Geometría Rentable en M15 ---")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv(config.DATA_RAW, sep='\t')
        if len(df.columns) < 2: df = pd.read_csv(config.DATA_RAW, sep=',')
    except:
        print("❌ Error: No encuentro el archivo de datos.")
        return

    # Limpieza básica
    col_map = {}
    for col in df.columns:
        c = col.upper().replace('<', '').replace('>', '')
        if 'CLOSE' in c: col_map[col] = 'Close'
        elif 'HIGH' in c: col_map[col] = 'High'
        elif 'LOW' in c: col_map[col] = 'Low'
    df = df.rename(columns=col_map)
    df = df[['High', 'Low', 'Close']].astype(float)
    
    # Calcular ATR
    df['PrevClose'] = df['Close'].shift(1)
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['PrevClose']), 
                                     abs(df['Low'] - df['PrevClose'])))
    df['ATR'] = df['TR'].rolling(14).mean()
    df.dropna(inplace=True)
    
    print(f"📊 Analizando {len(df)} velas M15...")
    
    # 2. Grid Search de Ratios (Fuerza Bruta)
    # Probamos combinaciones de SL y TP (multiplicadores de ATR)
    ratios = [
        (1.0, 1.0), # 1:1 Scalping equilibrado
        (1.0, 1.5), # 1:1.5 Ligera ventaja
        (1.0, 2.0), # 1:2 Trend Following
        (1.5, 1.5), # 1:1 Con más aire
        (0.5, 0.5), # Scalping ultra-rápido (HFT)
        (0.5, 1.0), # Sniper Scalping
        (2.0, 4.0)  # Swing Trading agresivo
    ]
    
    best_score = -9999
    best_config = None
    
    print(f"\n{'SL(ATR)':<8} | {'TP(ATR)':<8} | {'WIN RATE':<10} | {'EV (Esp. Mat)':<12} | {'CALIDAD'}")
    print("-" * 65)
    
    for sl_mult, tp_mult in ratios:
        wins = 0
        losses = 0
        
        # Simulación Vectorizada Rápida (Aprox) sobre 5000 velas aleatorias para velocidad
        # O sobre todo el dataset si es rápido
        sample_idxs = np.linspace(0, len(df)-200, 5000, dtype=int)
        
        for i in sample_idxs:
            entry = df['Close'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
            
            # Asumimos COMPRA aleatoria para ver la estructura del mercado
            # Si el mercado tiene sesgo alcista/bajista se notará, pero buscamos volatilidad
            # Chequeamos las siguientes 20 velas
            future_highs = df['High'].iloc[i+1:i+21].values
            future_lows = df['Low'].iloc[i+1:i+21].values
            
            hit_tp = False
            hit_sl = False
            
            for h, l in zip(future_highs, future_lows):
                if l <= (entry - sl_dist): hit_sl = True
                if h >= (entry + tp_dist): hit_tp = True
                
                if hit_sl and hit_tp: 
                    losses += 1 # Pesimismo: SL primero
                    break
                elif hit_sl:
                    losses += 1
                    break
                elif hit_tp:
                    wins += 1
                    break
        
        total = wins + losses
        if total == 0: continue
        
        win_rate = (wins / total) * 100
        # Esperanza Matemática por trade (en unidades de ATR)
        # EV = (ProbWin * Reward) - (ProbLoss * Risk)
        ev = ((win_rate/100) * tp_mult) - ((1 - (win_rate/100)) * sl_mult)
        
        calidad = "💀"
        if ev > 0: calidad = "✅"
        if ev > 0.1: calidad = "🔥 GEM"
        
        print(f"{sl_mult:<8} | {tp_mult:<8} | {win_rate:<9.1f}% | {ev:<12.3f} | {calidad}")
        
        if ev > best_score:
            best_score = ev
            best_config = (sl_mult, tp_mult)

    print("-" * 65)
    print(f"💡 RECOMENDACIÓN TÉCNICA: Usa SL={best_config[0]} ATR y TP={best_config[1]} ATR")
    print(f"   (Esto alinea tu bot con la realidad física del Oro)")

if __name__ == "__main__":
    auditar_ratios()