"""
Phoenix Parameter Optimizer - Grid Search optimizado para encontrar la mejor estrategia
Busca combinaciones de:
- UMBRAL_CONFIANZA: Qué tan seguro debe estar el modelo
- ATR_SL_MULTIPLIER: Dónde colocar el Stop Loss
- ATR_TP_MULTIPLIER: Dónde colocar el Take Profit
"""

import phoenix_config as config

if config.MODEL_TYPE == "xgboost":
    raise SystemExit("Parameter optimizer LSTM no aplica en modo XGBoost.")

import torch
import numpy as np
import pandas as pd
import joblib
import warnings
from itertools import product
from phoenix_brain import PhoenixLSTM
from phoenix_processor import PhoenixDataProcessor
from phoenix_metrics import TradingMetrics
import phoenix_config as config

warnings.filterwarnings('ignore')

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ParameterOptimizer:
    def __init__(self, model_path, scaler_path, df_test):
        self.model = PhoenixLSTM(input_size=config.INPUT_SIZE, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        
        self.scaler = joblib.load(scaler_path)
        self.df_test = df_test.reset_index(drop=True)
    
    def evaluar_combinacion(self, umbral, sl_mult, tp_mult):
        """
        Simula una estrategia con parámetros específicos
        Retorna métricas de performance
        """
        capital = config.CAPITAL_INICIAL
        capital_history = [capital]
        trades_log = []
        trades_count = 0
        
        # Preparar datos
        data_scaled = self.scaler.transform(self.df_test[config.FEATURES].values)
        tensor_test = torch.tensor(data_scaled, dtype=torch.float32).to(DEVICE)
        
        for i in range(config.LOOKBACK_WINDOW, len(self.df_test) - 20):
            if capital < config.CAPITAL_PROTECCIÓN:
                break
            
            # Filtro horario
            try:
                h = self.df_test.index[i].hour if hasattr(self.df_test.index[i], 'hour') else pd.Timestamp(self.df_test.index[i]).hour
            except:
                h = 12  # Default
            
            if h < config.HORA_INICIO or h >= config.HORA_CIERRE:
                continue
            
            # Inferencia
            window = tensor_test[i-config.LOOKBACK_WINDOW:i].unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(window)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                conf, pred = conf.item(), pred.item()
            
            # Filtro de confianza y volatilidad
            if pred != 0 and conf > umbral:
                price = self.df_test['Close'].iloc[i]
                atr = self.df_test['NATR'].iloc[i] * price / 100
                
                if atr < config.MIN_ATR_THRESHOLD:
                    continue
                
                # Gestión de riesgo mejorada
                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult
                
                riesgo_usd = capital * config.RIESGO_POR_OPERACION
                lotes = max(riesgo_usd / (sl_dist * 100), 0.01)
                
                # Precio objetivo
                if pred == 1:  # BUY
                    tp_price = price + tp_dist
                    sl_price = price - sl_dist
                else:  # SELL
                    tp_price = price - tp_dist
                    sl_price = price + sl_dist
                
                # Simulación intra-vela
                pnl = 0
                for j in range(1, 17):  # 4 horas
                    if i + j >= len(self.df_test):
                        break
                    
                    hi = self.df_test['High'].iloc[i+j]
                    lo = self.df_test['Low'].iloc[i+j]
                    
                    if pred == 1:  # BUY
                        if lo <= sl_price:
                            pnl = -sl_dist * 100 * lotes
                            break
                        if hi >= tp_price:
                            pnl = tp_dist * 100 * lotes
                            break
                    else:  # SELL
                        if hi >= sl_price:
                            pnl = -sl_dist * 100 * lotes
                            break
                        if lo <= tp_price:
                            pnl = tp_dist * 100 * lotes
                            break
                
                # Cierre por tiempo si no alcanzó objetivo
                if pnl == 0 and i + 16 < len(self.df_test):
                    exit_p = self.df_test['Close'].iloc[i+16]
                    pnl = (exit_p - price) * 100 * lotes if pred == 1 else (price - exit_p) * 100 * lotes
                
                # Comisión
                comision = (lotes / 0.01) * 0.06
                neto = pnl - comision
                
                capital += neto
                capital_history.append(capital)
                
                if neto > 0:
                    trades_log.append({
                        'entry_price': price,
                        'exit_price': price + (neto / (lotes * 100)) if pred == 1 else price - (neto / (lotes * 100)),
                        'direction': 'BUY' if pred == 1 else 'SELL',
                        'size': lotes,
                        'pnl': neto
                    })
                else:
                    trades_log.append({
                        'entry_price': price,
                        'exit_price': price - abs(neto / (lotes * 100)) if pred == 1 else price + abs(neto / (lotes * 100)),
                        'direction': 'BUY' if pred == 1 else 'SELL',
                        'size': lotes,
                        'pnl': neto
                    })
        
        # Calcular métricas
        if len(trades_log) == 0:
            return {
                'umbral': umbral,
                'sl_mult': sl_mult,
                'tp_mult': tp_mult,
                'capital_final': capital,
                'profit': capital - config.CAPITAL_INICIAL,
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe': 0,
                'calmar': 0,
                'score': -999
            }
        
        metrics = TradingMetrics(config.CAPITAL_INICIAL, trades_log)
        metricas = metrics.generar_reporte_completo(np.array(capital_history))
        
        # Score compuesto (peso hacia Sharpe y Win Rate)
        score = (
            metricas['sharpe_ratio'] * 0.4 +
            (metricas['win_rate'] / 100) * 0.3 +
            metricas['calmar_ratio'] * 0.2 +
            min(metricas['profit_factor'], 5) * 0.1
        )
        
        return {
            'umbral': umbral,
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'capital_final': capital,
            'profit': capital - config.CAPITAL_INICIAL,
            'profit_pct': ((capital - config.CAPITAL_INICIAL) / config.CAPITAL_INICIAL) * 100,
            'trades': metricas['total_trades'],
            'win_rate': metricas['win_rate'],
            'profit_factor': metricas['profit_factor'],
            'sharpe': metricas['sharpe_ratio'],
            'calmar': metricas['calmar_ratio'],
            'max_consecutive_wins': metricas['max_consecutive_wins'],
            'max_consecutive_losses': metricas['max_consecutive_losses'],
            'score': score
        }


def ejecutar_optimizador():
    """Grid search exhaustivo sobre parámetros clave"""
    
    print(f"\n{'='*80}")
    print(f" PHOENIX PARAMETER OPTIMIZER - Búsqueda de Configuración Óptima")
    print(f"{'='*80}")
    
    # Cargar datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    # Test set (último 30%)
    split_idx = int(len(df) * 0.7)
    df_test = df.iloc[split_idx:].copy()
    
    # Crear optimizador
    optimizer = ParameterOptimizer(
        config.MODEL_SAVE_PATH,
        config.SCALER_SAVE_PATH,
        df_test
    )
    
    # Grid search
    umbrales = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sl_mults = [1.0, 1.5, 2.0, 2.5]
    tp_mults = [1.5, 2.0, 2.5, 3.0]
    
    resultados = []
    total_combos = len(umbrales) * len(sl_mults) * len(tp_mults)
    
    print(f"\n🔍 Evaluando {total_combos} combinaciones...")
    print(f"\n{'CONF':<6} | {'SL':<5} | {'TP':<5} | {'PROFIT':<10} | {'TRADES':<7} | "
          f"{'WR %':<8} | {'PF':<6} | {'SHARPE':<8} | {'CALMAR':<8} | {'SCORE':<8}")
    print("-" * 110)
    
    combo = 0
    for umbral, sl_mult, tp_mult in product(umbrales, sl_mults, tp_mults):
        resultado = optimizer.evaluar_combinacion(umbral, sl_mult, tp_mult)
        resultados.append(resultado)
        
        combo += 1
        if combo % 10 == 0:
            print(f"{resultado['umbral']:<6.2f} | {resultado['sl_mult']:<5.1f} | "
                  f"{resultado['tp_mult']:<5.1f} | ${resultado['profit']:<9.2f} | "
                  f"{resultado['trades']:<7} | {resultado['win_rate']:<8.2f} | "
                  f"{resultado['profit_factor']:<6.2f} | {resultado['sharpe']:<8.2f} | "
                  f"{resultado['calmar']:<8.2f} | {resultado['score']:<8.2f}")
    
    # Ordenar por score
    resultados_sorted = sorted(resultados, key=lambda x: x['score'], reverse=True)
    
    print("-" * 110)
    print(f"\n{'='*80}")
    print(f" TOP 5 MEJORES CONFIGURACIONES")
    print(f"{'='*80}\n")
    
    for i, r in enumerate(resultados_sorted[:5], 1):
        print(f"{i}. UMBRAL={r['umbral']:.2f} | SL={r['sl_mult']:.1f} | TP={r['tp_mult']:.1f}")
        print(f"   Capital: ${r['capital_final']:.2f} | Profit: ${r['profit']:.2f} ({r['profit_pct']:.2f}%)")
        print(f"   Operaciones: {r['trades']} | Win Rate: {r['win_rate']:.2f}%")
        print(f"   Sharpe: {r['sharpe']:.2f} | Calmar: {r['calmar']:.2f} | Score: {r['score']:.4f}")
        print()
    
    # Guardar resultados
    df_resultados = pd.DataFrame(resultados_sorted)
    df_resultados.to_csv('optimization_results.csv', index=False)
    print(f"✅ Resultados guardados en 'optimization_results.csv'")
    
    # Actualizar config con mejor resultado
    mejor = resultados_sorted[0]
    print(f"\n💡 RECOMENDACIÓN:")
    print(f"   Actualiza phoenix_config.py con:")
    print(f"   UMBRAL_CONFIANZA = {mejor['umbral']:.2f}")
    print(f"   ATR_SL_MULTIPLIER = {mejor['sl_mult']:.1f}")
    print(f"   ATR_TP_MULTIPLIER = {mejor['tp_mult']:.1f}")


if __name__ == "__main__":
    ejecutar_optimizador()
