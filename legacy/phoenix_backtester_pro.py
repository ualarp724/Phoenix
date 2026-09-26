"""
Phoenix Professional Backtester
- Walk-Forward Analysis
- Stress Testing
- Métricas detalladas
- Detección de overfitting
"""

import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from phoenix_processor import PhoenixDataProcessor
from phoenix_metrics import TradingMetrics
from core.data_validators import validate_features, validate_no_nulls, validate_target
from core.execution_simulator import apply_execution_costs
from core.logging_utils import log_event
from core.regime import entry_allowed
from core.mtf import add_mtf_features_multi, mtf_confirm
from monitoring.metrics_store import log_metrics
from database.trade_store import init_db, insert_many, TradeRecord
from core.risk_manager import RiskManager
from core.execution_engine_stub import ExecutionEngine, OrderRequest
import phoenix_config as config
from core.news_checker import NewsFilter

warnings.filterwarnings('ignore')

class ProfessionalBacktester:
    def __init__(self, model_path, scaler_path):
        self.model_type = config.MODEL_TYPE
        self.model = joblib.load(model_path)
        self.model_lgbm = None
        if self.model_type == "ensemble":
            try:
                self.model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
            except Exception as exc:
                raise RuntimeError(f"No se pudo cargar LightGBM: {exc}")
        self.scaler = joblib.load(scaler_path)
        self.news_filter = NewsFilter()
    
    def backtest_period(
        self,
        df_period,
        umbral=None,
        sl_mult=None,
        tp_mult=None,
        precomputed_mtf=False,
        use_time_filter=None,
        hour_start=None,
        hour_end=None,
    ):
        """
        Backtea un período específico
        """
        umbral = umbral or config.UMBRAL_CONFIANZA
        sl_mult = sl_mult or config.ATR_SL_MULTIPLIER
        tp_mult = tp_mult or config.ATR_TP_MULTIPLIER
        
        init_db()
        capital = config.CAPITAL_INICIAL
        risk_manager = RiskManager(account_balance=capital)
        exec_engine = ExecutionEngine(risk_manager)
        capital_history = [capital]
        trades_log = []
        peak_capital = capital
        
        if not precomputed_mtf:
            df_period = add_mtf_features_multi(df_period.copy(), config.MTF_CONFIGS)
        df_period = df_period.dropna()
        validate_features(df_period)
        validate_no_nulls(df_period)
        validate_target(df_period)
        data_scaled = self.scaler.transform(df_period[config.FEATURES].values)

        lookback = config.LOOKBACK_WINDOW
        if len(df_period) <= lookback + 20:
            return None
        batch_size = getattr(config, "PRED_BATCH_SIZE", 512)
        windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
        total_windows = windows.shape[0]
        flat = windows.reshape(total_windows, -1)
        predictions = np.empty(total_windows, dtype=np.int64)
        confidences = np.empty(total_windows, dtype=np.float32)
        for start in range(0, total_windows, batch_size):
            end = min(start + batch_size, total_windows)
            if self.model_type == "ensemble" and self.model_lgbm is not None:
                probs_xgb = self.model.predict_proba(flat[start:end])
                probs_lgb = self.model_lgbm.predict_proba(flat[start:end])
                weights = config.ENSEMBLE_WEIGHTS

                if config.ENSEMBLE_REQUIRE_CONSENSUS:
                    preds_xgb = np.argmax(probs_xgb, axis=1)
                    preds_lgb = np.argmax(probs_lgb, axis=1)
                    confs_xgb = np.max(probs_xgb, axis=1)
                    confs_lgb = np.max(probs_lgb, axis=1)
                    consensus = (preds_xgb == preds_lgb) & (preds_xgb != 0)
                    preds = np.where(consensus, preds_xgb, 0)
                    confs = np.where(
                        consensus,
                        (weights["xgboost"] * confs_xgb) + (weights["lightgbm"] * confs_lgb),
                        0.0,
                    )
                else:
                    combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)
                    buy_scores = combined[:, 1]
                    sell_scores = combined[:, 2]
                    preds = np.where(
                        buy_scores >= sell_scores,
                        np.where(buy_scores >= config.UMBRAL_BUY, 1, 0),
                        np.where(sell_scores >= config.UMBRAL_SELL, 2, 0),
                    )
                    confs = np.where(preds == 1, buy_scores, np.where(preds == 2, sell_scores, 0.0))
            else:
                probs = self.model.predict_proba(flat[start:end])
                preds = np.argmax(probs, axis=1)
                confs = np.max(probs, axis=1)

            predictions[start:end] = preds
            confidences[start:end] = confs

        conf_np = confidences[confidences > 0]
        if conf_np.size == 0:
            conf_np = confidences
        dyn_threshold = float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))

        for i in range(total_windows - 20):
            if capital < config.CAPITAL_PROTECCIÓN:
                break

            actual_idx = i + lookback

            # Circuit breaker: max drawdown 20%
            if capital > peak_capital:
                peak_capital = capital
            max_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if max_dd >= 0.20:
                break
            
            # Filtro horario
            try:
                current_ts = df_period.index[actual_idx]
                h = current_ts.hour if hasattr(current_ts, 'hour') else 12
            except Exception:
                current_ts = None
                h = 12
            
            time_filter = config.USE_TIME_FILTER if use_time_filter is None else use_time_filter
            start_h = config.HORA_INICIO if hour_start is None else hour_start
            end_h = config.HORA_CIERRE if hour_end is None else hour_end
            if time_filter and (h < start_h or h >= end_h):
                continue

            if current_ts is not None and self.news_filter.high_impact_at(current_ts):
                continue
            
            # Inferencia
            conf = float(confidences[i])
            pred = int(predictions[i])
            
            # Filtros
            if pred != 0 and conf > max(umbral, dyn_threshold):
                row = df_period.iloc[actual_idx]
                price = row['Close']
                atr = df_period['NATR'].iloc[actual_idx] * price / 100

                # Filtro mechas (wick) usando cuerpo vs cierre previo
                if actual_idx > 0:
                    prev_close = df_period['Close'].iloc[actual_idx - 1]
                    body = abs(price - prev_close)
                    wick = max(row['High'] - row['Low'] - body, 0.0)
                    if body == 0 or wick > (0.5 * body):
                        continue

                # Flash crash: ATR 3 velas > 200% del promedio diario
                atr_series = (df_period['NATR'] * df_period['Close'] / 100.0)
                atr_last3 = atr_series.iloc[max(actual_idx - 2, 0): actual_idx + 1].mean()
                atr_daily_avg = atr_series.iloc[max(actual_idx - 288, 0): actual_idx + 1].mean()
                if risk_manager.block_flash_crash(atr_last3, atr_daily_avg):
                    continue

                if row['Vol_Rel'] < config.MIN_VOL_REL:
                    continue
                if config.USE_REGIME_FILTER and not entry_allowed(pred, row):
                    continue
                if config.USE_M15_FILTER and config.MTF_ENFORCE_TREND:
                    m15_trend = float(row.get("M15_Trend_Score", 0.0))
                    if pred == 1 and m15_trend < config.M15_TREND_MIN:
                        continue
                    if pred == 2 and m15_trend > -config.M15_TREND_MIN:
                        continue
                if config.USE_MTF_CONFIRM and not mtf_confirm(pred, row, prefix="H1_"):
                    continue
                
                if atr < config.MIN_ATR_THRESHOLD:
                    continue
                
                # Risk management
                sl_dist = atr * sl_mult
                tp_dist = atr * tp_mult
                
                target_mult = (config.TARGET_DAILY_USD / config.EXPECTED_DAILY_USD) if config.EXPECTED_DAILY_USD else 1.0
                riesgo_pct = config.RIESGO_POR_OPERACION * target_mult * config.RISK_MULTIPLIER
                riesgo_pct = min(config.MAX_RISK_PER_TRADE_PCT, riesgo_pct)
                riesgo = capital * riesgo_pct
                lotes = max(riesgo / (sl_dist * 100), 0.01)
                lotes = max(config.MIN_LOT_SIZE, min(lotes, config.MAX_LOT_SIZE))

                risk_manager.account_balance = capital
                order_id = exec_engine.submit_order(
                    OrderRequest(
                        symbol=config.SYMBOL,
                        side='BUY' if pred == 1 else 'SELL',
                        lot_size=lotes,
                        stop_loss_pips=config.DEFAULT_STOP_LOSS_PIPS,
                        take_profit_pips=max(
                            config.MIN_TAKE_PROFIT_PIPS,
                            int(config.DEFAULT_STOP_LOSS_PIPS * config.RISK_REWARD_RATIO_TARGET),
                        ),
                        timestamp=current_ts,
                    )
                )
                if not order_id:
                    continue
                
                # Precios
                if pred == 1:  # BUY
                    tp_price = price + tp_dist
                    sl_price = price - sl_dist
                else:  # SELL
                    tp_price = price - tp_dist
                    sl_price = price + sl_dist
                
                # Simulación
                pnl = 0
                for j in range(1, 17):
                    if actual_idx + j >= len(df_period):
                        break
                    
                    hi = df_period['High'].iloc[actual_idx + j]
                    lo = df_period['Low'].iloc[actual_idx + j]
                    
                    if pred == 1:
                        if lo <= sl_price:
                            pnl = -sl_dist * 100 * lotes
                            break
                        if hi >= tp_price:
                            pnl = tp_dist * 100 * lotes
                            break
                    else:
                        if hi >= sl_price:
                            pnl = -sl_dist * 100 * lotes
                            break
                        if lo <= tp_price:
                            pnl = tp_dist * 100 * lotes
                            break
                
                # Timeout
                if pnl == 0 and actual_idx + 16 < len(df_period):
                    exit_p = df_period['Close'].iloc[actual_idx + 16]
                    pnl = (exit_p - price) * 100 * lotes if pred == 1 else (price - exit_p) * 100 * lotes
                
                # Comisión + ejecución (spread/slippage)
                exec_side = 'BUY' if pred == 1 else 'SELL'
                exec_result = apply_execution_costs(price, lotes, exec_side)
                comision = (lotes / 0.01) * 0.06
                neto = pnl - comision - exec_result.spread_usd - exec_result.slippage_usd
                
                capital += neto
                risk_manager.record_trade_closed(neto)
                capital_history.append(capital)
                
                trades_log.append({
                    'entry_price': price,
                    'exit_price': price,
                    'direction': 'BUY' if pred == 1 else 'SELL',
                    'size': lotes,
                    'pnl': neto,
                    'timestamp': current_ts
                })
                log_event({
                    "event": "BACKTEST_TRADE",
                    "direction": 'BUY' if pred == 1 else 'SELL',
                    "pnl": neto,
                    "sl_dist": sl_dist,
                    "tp_dist": tp_dist,
                })
        
        # Métricas
        if len(trades_log) == 0:
            return None
        
        metrics = TradingMetrics(config.CAPITAL_INICIAL, trades_log)
        summary = {
            'capital_final': capital,
            'profit': capital - config.CAPITAL_INICIAL,
            'trades': len(trades_log),
            'capital_history': capital_history,
            'trades_log': trades_log,
            'metrics': metrics.generar_reporte_completo(np.array(capital_history))
        }
        log_metrics(summary['metrics'])
        insert_many(
            TradeRecord(
                timestamp=t.get('timestamp') or datetime.utcnow(),
                symbol=config.SYMBOL,
                side=t['direction'],
                lot_size=t['size'],
                entry_price=t['entry_price'],
                exit_price=t['exit_price'],
                pnl=t['pnl'],
            )
            for t in trades_log
        )
        return summary

    def backtest_date_range(self, df, start_date, end_date, **kwargs):
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None
        subset = df.copy()
        if start_ts is not None:
            subset = subset[subset.index >= start_ts]
        if end_ts is not None:
            subset = subset[subset.index <= end_ts]
        if subset.empty:
            return None
        return self.backtest_period(subset, **kwargs)

    def backtest_splits(self, df, **kwargs):
        results = {}
        results["train"] = self.backtest_date_range(
            df,
            getattr(config, "TRAIN_START_DATE", None),
            getattr(config, "TRAIN_END_DATE", None),
            **kwargs,
        )
        results["val"] = self.backtest_date_range(
            df,
            getattr(config, "VAL_START_DATE", None),
            getattr(config, "VAL_END_DATE", None),
            **kwargs,
        )
        results["test"] = self.backtest_date_range(
            df,
            getattr(config, "TEST_START_DATE", None),
            getattr(config, "TEST_END_DATE", None),
            **kwargs,
        )
        return results
    
    def walk_forward_analysis(self, df, window_size_pct=20, step_size_pct=10):
        """
        Walk-Forward Analysis para detectar overfitting
        Divide datos en ventanas deslizantes
        """
        n_rows = len(df)
        window_size = int(n_rows * window_size_pct / 100)
        step_size = int(n_rows * step_size_pct / 100)
        
        results = []
        start_idx = 0
        
        print(f"\n🔄 Walk-Forward Analysis ({window_size} velas por ventana, paso de {step_size})")
        print(f"{'VENTANA':<10} | {'PROFIT':<10} | {'TRADES':<7} | {'WIN%':<8} | {'SHARPE':<8}")
        print("-" * 55)
        
        window_num = 1
        while start_idx + window_size < n_rows:
            end_idx = start_idx + window_size
            df_window = df.iloc[start_idx:end_idx]
            
            result = self.backtest_period(df_window)
            if result:
                result['window'] = window_num
                result['start_idx'] = start_idx
                result['end_idx'] = end_idx
                results.append(result)
                
                metrics = result['metrics']
                print(f"{window_num:<10} | ${result['profit']:<9.2f} | "
                      f"{metrics['total_trades']:<7} | {metrics['win_rate']:<8.2f} | "
                      f"{metrics['sharpe_ratio']:<8.2f}")
            
            start_idx += step_size
            window_num += 1
        
        return results
    
    def stress_test_slippage(self, df, slippage_pct_list=[0, 0.1, 0.2, 0.5]):
        """
        Stress test: Simula impacto de slippage (diferencia de precio)
        """
        print(f"\n⚡ Stress Test - Impacto de Slippage")
        print(f"{'SLIPPAGE %':<12} | {'PROFIT':<12} | {'WIN RATE':<10} | {'SHARPE':<8}")
        print("-" * 50)
        
        base_result = self.backtest_period(df)
        if not base_result:
            print("❌ Backtesting falló")
            return
        
        for slippage_pct in slippage_pct_list:
            # Aplicar slippage a los trades
            trades_modified = []
            for trade in base_result['trades_log']:
                trade_modified = trade.copy()
                # Slippage desfavorable
                slippage_usd = trade['entry_price'] * trade['size'] * 100 * (slippage_pct / 100)
                trade_modified['pnl'] -= slippage_usd
                trades_modified.append(trade_modified)
            
            metrics = TradingMetrics(config.CAPITAL_INICIAL, trades_modified)
            metricas = metrics.generar_reporte_completo(np.array(base_result['capital_history']))
            
            capital_final = config.CAPITAL_INICIAL + sum([t['pnl'] for t in trades_modified])
            profit = capital_final - config.CAPITAL_INICIAL
            
            print(f"{slippage_pct:<12.2f} | ${profit:<11.2f} | {metricas['win_rate']:<10.2f} | "
                  f"{metricas['sharpe_ratio']:<8.2f}")


def ejecutar_backtest_profesional():
    print(f"\n{'='*80}")
    print(f" PHOENIX PROFESSIONAL BACKTESTER")
    print(f"{'='*80}")
    
    # Cargar datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = add_mtf_features_multi(processor.clean_and_prepare(apply_train_filter=False), config.MTF_CONFIGS)
    
    # Crear backtester
    backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    
    # 1. Backtesting normal
    print(f"\n📊 Backtesting completo...")
    df_test = df.iloc[int(len(df) * 0.7):].copy()
    result = backtester.backtest_period(df_test)
    
    if result:
        metrics = result['metrics']
        print(f"\n{'='*80}")
        print(f" RESULTADOS DEL BACKTEST")
        print(f"{'='*80}")
        print(f"Capital Final:         ${result['capital_final']:.2f}")
        print(f"Profit:                ${result['profit']:.2f}")
        print(f"Total Operaciones:     {metrics['total_trades']}")
        print(f"Ganadas:               {metrics['winning_trades']}")
        print(f"Perdidas:              {metrics['losing_trades']}")
        print(f"Win Rate:              {metrics['win_rate']:.2f}%")
        print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
        print(f"\nSharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:.2f}")
        print(f"Recovery Factor:       {metrics['recovery_factor']:.2f}")
        print(f"Max Consecutive Wins:  {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Loss:  {metrics['max_consecutive_losses']}")
        print(f"{'='*80}\n")
    
    # 2. Walk-Forward Analysis
    results_wf = backtester.walk_forward_analysis(df_test)
    
    # 3. Stress Test
    backtester.stress_test_slippage(df_test)
    
    print(f"\n✅ Backtesting completado")


if __name__ == "__main__":
    ejecutar_backtest_profesional()
