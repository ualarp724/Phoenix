#!/usr/bin/env python3
"""
BTC Forensic Report (last 4 months: rows 66000-99939)
Prints each trade and summary metrics.
"""

import copy
from datetime import datetime

import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from phoenix_backtester_pro import TradingMetrics
from core.mtf import add_mtf_features_multi
from core.execution_simulator import apply_execution_costs
from core.risk_manager import RiskManager

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")

from sklearn.preprocessing import StandardScaler

# Params requested
UMBRAL_BUY = 0.216368
UMBRAL_SELL = 0.239545
SL_MULT = 2.401460
TP_MULT = 2.066247

BEST_PARAMS = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.061263597364696325,
    "min_child_weight": 22,
    "gamma": 0.18478745296914784,
    "subsample": 0.6920408477000773,
    "colsample_bytree": 0.9136472193306594,
    "reg_alpha": 0.4016246397170869,
    "reg_lambda": 2.759602133276472,
    "max_delta_step": 1,
}


def main() -> None:
    config.apply_asset("BTCUSD")
    config.ENSEMBLE_REQUIRE_CONSENSUS = False

    config.UMBRAL_BUY = UMBRAL_BUY
    config.UMBRAL_SELL = UMBRAL_SELL
    config.ATR_SL_MULTIPLIER = SL_MULT
    config.ATR_TP_MULTIPLIER = TP_MULT

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.4),
        config.MTF_CONFIGS,
    )
    for col in config.FEATURES:
        if col not in df_all.columns:
            df_all[col] = 0.0

    df_all = df_all.dropna()

    train_df = df_all.iloc[:66000].copy()
    val_df = df_all.iloc[66000:99939].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    params_xgb = copy.deepcopy(config.XGB_PARAMS)
    params_xgb.update(BEST_PARAMS)
    params_xgb.setdefault("n_jobs", 1)
    params_xgb.setdefault("tree_method", "hist")

    xgb_model = xgb.XGBClassifier(**params_xgb)
    xgb_model.fit(X_train, y_train, verbose=False)

    lgbm_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    lgbm_model.fit(X_train, y_train)

    # --- Backtest val_df with detailed trade logging ---
    capital = config.CAPITAL_INICIAL
    risk_manager = RiskManager(account_balance=capital)
    capital_history = [capital]
    trades_log = []
    opportunities = 0

    data_scaled = scaler.transform(val_df[config.FEATURES].values)
    lookback = config.LOOKBACK_WINDOW
    if len(val_df) <= lookback + 20:
        raise SystemExit("Datos insuficientes en val_df.")

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)

    probs_xgb = xgb_model.predict_proba(flat)
    probs_lgb = lgbm_model.predict_proba(flat)
    weights = config.ENSEMBLE_WEIGHTS

    combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)
    buy_scores = combined[:, 1]
    sell_scores = combined[:, 2]
    preds = np.where(
        buy_scores >= sell_scores,
        np.where(buy_scores >= config.UMBRAL_BUY, 1, 0),
        np.where(sell_scores >= config.UMBRAL_SELL, 2, 0),
    )
    confs = np.where(preds == 1, buy_scores, np.where(preds == 2, sell_scores, 0.0))

    conf_np = confs[confs > 0]
    if conf_np.size == 0:
        conf_np = confs
    dyn_threshold = float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))

    print("FechaHora\tTipo\tEntrada\tSL\tTP\tPnL_USD")

    for i in range(total_windows - 20):
        if capital < config.CAPITAL_PROTECCIÓN:
            break

        actual_idx = i + lookback
        row = val_df.iloc[actual_idx]
        price = row['Close']

        # Confidence gate
        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf <= max(config.UMBRAL_CONFIANZA, dyn_threshold):
            continue

        # Filters
        if actual_idx > 0:
            prev_close = val_df['Close'].iloc[actual_idx - 1]
            body = abs(price - prev_close)
            wick = max(row['High'] - row['Low'] - body, 0.0)
            if body == 0 or wick > (0.5 * body):
                continue

        atr_series = (val_df['NATR'] * val_df['Close'] / 100.0)
        atr_last3 = atr_series.iloc[max(actual_idx - 2, 0): actual_idx + 1].mean()
        atr_daily_avg = atr_series.iloc[max(actual_idx - 288, 0): actual_idx + 1].mean()
        if risk_manager.block_flash_crash(atr_last3, atr_daily_avg):
            continue

        if row['Vol_Rel'] < config.MIN_VOL_REL:
            continue

        if config.USE_REGIME_FILTER:
            from core.regime import entry_allowed
            if not entry_allowed(pred, row):
                continue

        if config.USE_M15_FILTER and config.MTF_ENFORCE_TREND:
            m15_trend = float(row.get("M15_Trend_Score", 0.0))
            if pred == 1 and m15_trend < config.M15_TREND_MIN:
                continue
            if pred == 2 and m15_trend > -config.M15_TREND_MIN:
                continue

        if config.USE_MTF_CONFIRM:
            from core.mtf import mtf_confirm
            if not mtf_confirm(pred, row, prefix="H1_"):
                continue

        atr = val_df['NATR'].iloc[actual_idx] * price / 100
        if atr < config.MIN_ATR_THRESHOLD:
            continue

        opportunities += 1

        sl_dist = atr * config.ATR_SL_MULTIPLIER
        tp_dist = atr * config.ATR_TP_MULTIPLIER

        target_mult = (config.TARGET_DAILY_USD / config.EXPECTED_DAILY_USD) if config.EXPECTED_DAILY_USD else 1.0
        riesgo_pct = config.RIESGO_POR_OPERACION * target_mult * config.RISK_MULTIPLIER
        riesgo_pct = min(config.MAX_RISK_PER_TRADE_PCT, riesgo_pct)
        riesgo = capital * riesgo_pct
        lotes = max(riesgo / (sl_dist * 100), 0.01)
        lotes = max(config.MIN_LOT_SIZE, min(lotes, config.MAX_LOT_SIZE))

        if pred == 1:
            tp_price = price + tp_dist
            sl_price = price - sl_dist
            side = "COMPRA"
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist
            side = "VENTA"

        pnl = 0.0
        for j in range(1, 17):
            if actual_idx + j >= len(val_df):
                break
            hi = val_df['High'].iloc[actual_idx + j]
            lo = val_df['Low'].iloc[actual_idx + j]
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

        if pnl == 0 and actual_idx + 16 < len(val_df):
            exit_p = val_df['Close'].iloc[actual_idx + 16]
            pnl = (exit_p - price) * 100 * lotes if pred == 1 else (price - exit_p) * 100 * lotes

        exec_result = apply_execution_costs(price, lotes, 'BUY' if pred == 1 else 'SELL')
        comision = (lotes / 0.01) * 0.06
        neto = pnl - comision - exec_result.spread_usd - exec_result.slippage_usd

        capital += neto
        capital_history.append(capital)

        trades_log.append({
            'entry_price': price,
            'exit_price': price,
            'direction': side,
            'size': lotes,
            'pnl': neto,
            'timestamp': row.name if hasattr(row, 'name') else datetime.utcnow(),
        })

        ts = row.name if hasattr(row, 'name') else datetime.utcnow()
        print(f"{ts}\t{side}\t{price:.2f}\t{sl_price:.2f}\t{tp_price:.2f}\t{neto:.2f}")

    if not trades_log:
        print("SIN TRADES")
        return

    metrics = TradingMetrics(config.CAPITAL_INICIAL, trades_log)
    summary = metrics.generar_reporte_completo(np.array(capital_history))

    print(f"OPORTUNIDADES: {opportunities}")
    print(f"PROFIT FACTOR: {summary.get('profit_factor', 0.0):.2f}")
    print(f"WIN RATE: {summary.get('win_rate', 0.0):.2f}%")
    print(f"MAX DRAWDOWN %: {summary.get('max_drawdown_pct', 0.0):.2f}")


if __name__ == "__main__":
    main()
