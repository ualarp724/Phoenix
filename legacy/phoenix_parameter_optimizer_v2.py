#!/usr/bin/env python3
"""
Phoenix Parameter Optimizer - Versión Rápida y Eficiente
Realiza grid search en 96 combinaciones de parámetros
"""

import phoenix_config as config

if config.MODEL_TYPE == "xgboost":
    raise SystemExit("Parameter optimizer v2 LSTM no aplica en modo XGBoost.")

import torch
import numpy as np
import pandas as pd
import joblib
import warnings
from itertools import product
from tqdm import tqdm
from phoenix_brain import PhoenixLSTM
from phoenix_processor import PhoenixDataProcessor
from phoenix_metrics import TradingMetrics
from core.data_validators import validate_features, validate_no_nulls, validate_target
from core.execution_simulator import apply_execution_costs
from core.logging_utils import log_event
from core.news_checker import NewsFilter
from core.calibration import apply_temperature, load_temperature
from core.regime import entry_allowed
from core.mtf import add_mtf_features, mtf_confirm
from monitoring.metrics_store import log_metrics
from database.trade_store import init_db

warnings.filterwarnings('ignore')

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class FastParameterOptimizer:
    def __init__(self, model_path, scaler_path):
        """Inicializa optimizer con modelo y scaler"""
        print(f"🚀 Inicializando optimizer...")
        print(f"   Device: {DEVICE}")
        
        # Cargar modelo
        self.model = PhoenixLSTM(
            input_size=config.INPUT_SIZE,
            hidden_layers=config.HIDDEN_LAYERS,
            num_classes=3
        )
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        print(f"   ✅ Modelo cargado")
        
        # Cargar scaler
        self.scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler cargado")
        self.news_filter = NewsFilter()
        
        # Cargar datos de test
        processor = PhoenixDataProcessor(config.DATA_RAW)
        df = add_mtf_features(processor.clean_and_prepare())
        train_size = int(0.7 * len(df))
        self.df_test = df.iloc[train_size:]
        validate_features(self.df_test)
        validate_no_nulls(self.df_test)
        validate_target(self.df_test)
        print(f"   ✅ Datos de test: {len(self.df_test)} velas")
    
    def evaluar_parametros(self, umbral, sl_mult, tp_mult, batch_size=None):
        """Evalúa una combinación de parámetros en backtest"""
        if batch_size is None:
            batch_size = getattr(config, "PRED_BATCH_SIZE", 1024)
        capital = config.CAPITAL_INICIAL
        capital_history = [capital]
        trades = []
        
        # Preparar datos escalados
        data_scaled = self.scaler.transform(self.df_test[config.FEATURES].values)
        tensor_data = torch.tensor(data_scaled, dtype=torch.float32).to(DEVICE)

        temperature = load_temperature()
        with torch.no_grad():
            # Generar predicciones por lotes (mucho más rápido en MPS)
            lookback = config.LOOKBACK_WINDOW
            if len(tensor_data) <= lookback:
                return None

            windows = tensor_data.unfold(0, lookback, 1).permute(0, 2, 1)  # [N-lookback+1, lookback, features]
            total_windows = windows.shape[0]
                predictions = torch.empty(total_windows, dtype=torch.int64)
                confidences = torch.empty(total_windows, dtype=torch.float32)

            for start in range(0, total_windows, batch_size):
                end = min(start + batch_size, total_windows)
                batch = windows[start:end]
                logits = apply_temperature(self.model(batch), temperature)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                predictions[start:end] = pred.cpu()
                confidences[start:end] = conf.cpu()

                # Umbral dinámico por percentil
                conf_np = confidences.numpy()
                dyn_threshold = float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))

                # Simular trading con los parámetros
            horizon = 32
            for i in range(total_windows):
                actual_idx = i + lookback
                if actual_idx + horizon >= len(self.df_test):
                    break

                row = self.df_test.iloc[actual_idx]
                price = row['Close']
                natr = self.df_test.iloc[actual_idx].get('NATR', 0.0)
                atr = (natr / 100.0) * price if natr > 0 else price * 0.002
                pred = int(predictions[i].item())
                conf = float(confidences[i].item())

                try:
                    current_ts = self.df_test.index[actual_idx]
                except Exception:
                    current_ts = None
                if config.USE_TIME_FILTER and current_ts is not None:
                    h = current_ts.hour if hasattr(current_ts, 'hour') else 12
                    if h < config.HORA_INICIO or h >= config.HORA_CIERRE:
                        continue
                if current_ts is not None and self.news_filter.high_impact_at(current_ts):
                    continue

                if row.get('Vol_Rel', 0.0) < config.MIN_VOL_REL:
                    continue
                if config.USE_REGIME_FILTER and not entry_allowed(pred, row):
                    continue
                if config.USE_MTF_CONFIRM and not mtf_confirm(pred, row):
                    continue

                    if conf < max(umbral, dyn_threshold) or pred == 0:
                    continue

                if capital < config.CAPITAL_PROTECCIÓN:
                    break

                sl = atr * sl_mult
                tp = atr * tp_mult
                if sl <= 0 or tp <= 0:
                    continue

                future_slice = self.df_test.iloc[actual_idx:actual_idx + horizon]
                future_high = future_slice['High'].max()
                future_low = future_slice['Low'].min()

                risk_usd = capital * config.MAX_RISK_PER_TRADE_PCT
                pnl = 0.0

                if pred == 1:  # BUY
                    tp_price = price + tp
                    sl_price = price - sl
                    hit_tp = future_high >= tp_price
                    hit_sl = future_low <= sl_price
                    if hit_tp and not hit_sl:
                        pnl = risk_usd * (tp / sl)
                    elif hit_sl:
                        pnl = -risk_usd
                    else:
                        continue
                elif pred == 2:  # SELL
                    tp_price = price - tp
                    sl_price = price + sl
                    hit_tp = future_low <= tp_price
                    hit_sl = future_high >= sl_price
                    if hit_tp and not hit_sl:
                        pnl = risk_usd * (tp / sl)
                    elif hit_sl:
                        pnl = -risk_usd
                    else:
                        continue

                exec_side = 'BUY' if pred == 1 else 'SELL'
                exec_result = apply_execution_costs(price, 0.01, exec_side)
                pnl = pnl - exec_result.spread_usd - exec_result.slippage_usd

                capital += pnl
                capital_history.append(max(capital, config.CAPITAL_PROTECCIÓN))
                trades.append({
                    'price': price,
                    'side': 'BUY' if pred == 1 else 'SELL',
                    'pnl': pnl,
                    'confidence': conf
                })
                log_event({
                    "event": "OPT_TRADE",
                    "direction": 'BUY' if pred == 1 else 'SELL',
                    "pnl": pnl,
                    "sl_mult": sl_mult,
                    "tp_mult": tp_mult,
                    "umbral": umbral,
                })
        
        # Calcular métricas
        if len(trades) > 0:
            metrics = TradingMetrics(config.CAPITAL_INICIAL, trades)
            return {
                'trades': len(trades),
                'final_capital': capital,
                'pnl': capital - config.CAPITAL_INICIAL,
                'pnl_pct': (capital - config.CAPITAL_INICIAL) / config.CAPITAL_INICIAL * 100,
                'capital_history': capital_history
            }
        else:
            return {
                'trades': 0,
                'final_capital': capital,
                'pnl': 0,
                'pnl_pct': 0,
                'capital_history': capital_history
            }
    
    def optimizar(self):
        """Ejecuta grid search en 96 combinaciones"""
        print(f"\n📊 Optimizando 96 combinaciones de parámetros...")
        init_db()
        
        # Definir grilla
        umbrales = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        sl_mults = [1.0, 1.5, 2.0]
        tp_mults = [1.5, 2.0, 2.5, 3.0]
        
        results = []
        total = len(umbrales) * len(sl_mults) * len(tp_mults)
        
        with tqdm(total=total, desc="Grid Search") as pbar:
            for umbral, sl_mult, tp_mult in product(umbrales, sl_mults, tp_mults):
                resultado = self.evaluar_parametros(umbral, sl_mult, tp_mult)
                resultado.update({
                    'umbral': umbral,
                    'sl_mult': sl_mult,
                    'tp_mult': tp_mult
                })
                results.append(resultado)
                pbar.update(1)
        
        # Convertir a DataFrame y ordenar por PnL
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('pnl', ascending=False)
        
        # Guardar resultados
        df_results.to_csv('optimization_results.csv', index=False)
        print(f"\n✅ Resultados guardados en optimization_results.csv")
        
        # Mostrar TOP 5
        print(f"\n🏆 TOP 5 MEJORES CONFIGURACIONES:")
        print("="*80)
        for idx, row in df_results.head(5).iterrows():
            print(f"\n{idx+1}. PnL: ${row['pnl']:,.2f} ({row['pnl_pct']:.2f}%)")
            print(f"   UMBRAL_CONFIANZA: {row['umbral']:.2f}")
            print(f"   ATR_SL_MULTIPLIER: {row['sl_mult']:.2f}")
            print(f"   ATR_TP_MULTIPLIER: {row['tp_mult']:.2f}")
            print(f"   Trades: {row['trades']}")
            print(f"   Capital Final: ${row['final_capital']:,.2f}")
        
        if not df_results.empty:
            best = df_results.iloc[0]
            log_metrics({
                "event": "OPTIMIZER_BEST",
                "pnl": float(best["pnl"]),
                "trades": int(best["trades"]),
                "umbral": float(best["umbral"]),
                "sl_mult": float(best["sl_mult"]),
                "tp_mult": float(best["tp_mult"]),
            })
        return df_results

if __name__ == "__main__":
    import os
    
    model_path = "phoenix_brain.pth"
    scaler_path = "phoenix_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        print(f"   Ejecuta primero: python3 phoenix_evolution.py")
        exit(1)
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler no encontrado: {scaler_path}")
        print(f"   Ejecuta primero: python3 phoenix_evolution.py")
        exit(1)
    
    print(f"\n{'='*80}")
    print(f" PHOENIX PARAMETER OPTIMIZER - Grid Search (96 Combinaciones)")
    print(f"{'='*80}")
    
    optimizer = FastParameterOptimizer(model_path, scaler_path)
    results = optimizer.optimizar()
    
    print(f"\n{'='*80}")
    print(f"✅ Optimización completada")
    print(f"{'='*80}\n")
