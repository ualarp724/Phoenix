#!/usr/bin/env python3
"""
Informe real BTC (Best Trial RR>=1:2)
- Entrena con primeras 66,000 filas
- Reporta trades en filas 66,000-99,000
- Sin filtros de élite (sin vol_rel, mechas, MTF, régimen, news)
- PnL: 1 punto = $1 por lote (sin *100)
"""

import copy
from datetime import datetime

import numpy as np

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")

from sklearn.preprocessing import StandardScaler

UMBRAL_BUY = 0.21520570453369903
UMBRAL_SELL = 0.10633549974478322
SL_MULT = 1.1596330355344473
TP_MULT = 5.160572367353982

BEST_PARAMS = {
    "max_depth": 5,
    "n_estimators": 550,
    "learning_rate": 0.05883428517344634,
    "min_child_weight": 11,
    "gamma": 0.2735594181394874,
    "subsample": 0.8093894689378842,
    "colsample_bytree": 0.8536095065346021,
    "reg_alpha": 0.35241336108374904,
    "reg_lambda": 2.417037176912407,
    "max_delta_step": 1,
}


def max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    return max_dd


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
    val_df = df_all.iloc[66000:99000].copy()

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

    capital = config.CAPITAL_INICIAL
    equity = [capital]
    wins = 0
    total_trades = 0
    esta_en_operacion = False
    loss_streak = 0
    max_loss_streak = 0
    total_lots = 0.0
    tp_hits = 0
    timeout_hits = 0

    print("Fecha\tTipo\tEntrada\tSalida\tPnL_USD\tDuración")

    i = 0
    while i < (total_windows - 12):
        actual_idx = i + lookback
        row = val_df.iloc[actual_idx]
        price = row['Close']

        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf <= max(config.UMBRAL_CONFIANZA, dyn_threshold):
            i += 1
            continue

        if esta_en_operacion:
            i += 1
            continue

        atr = val_df['NATR'].iloc[actual_idx] * price / 100
        if atr < config.MIN_ATR_THRESHOLD:
            i += 1
            continue

        sl_dist = atr * config.ATR_SL_MULTIPLIER
        tp_dist = atr * config.ATR_TP_MULTIPLIER

        lotes = 3.0 / max(tp_dist, 1e-9)
        lotes = max(config.MIN_LOT_SIZE, min(lotes, config.MAX_LOT_SIZE))
        if sl_dist * lotes > 1.50:
            i += 1
            continue

        if pred == 1:
            tp_price = price + tp_dist
            sl_price = price - sl_dist
            side = "COMPRA"
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist
            side = "VENTA"

        pnl = 0.0
        exit_price = price
        exit_idx = None
        exit_reason = "TIMEOUT"
        esta_en_operacion = True
        for j in range(1, 13):
            if actual_idx + j >= len(val_df):
                break
            hi = val_df['High'].iloc[actual_idx + j]
            lo = val_df['Low'].iloc[actual_idx + j]
            if pred == 1:
                if lo <= sl_price:
                    pnl = -abs(price - sl_price) * lotes
                    exit_price = sl_price
                    exit_idx = actual_idx + j
                    exit_reason = "SL"
                    break
                if hi >= tp_price:
                    pnl = abs(tp_price - price) * lotes
                    exit_price = tp_price
                    exit_idx = actual_idx + j
                    exit_reason = "TP"
                    break
            else:
                if hi >= sl_price:
                    pnl = -abs(sl_price - price) * lotes
                    exit_price = sl_price
                    exit_idx = actual_idx + j
                    exit_reason = "SL"
                    break
                if lo <= tp_price:
                    pnl = abs(price - tp_price) * lotes
                    exit_price = tp_price
                    exit_idx = actual_idx + j
                    exit_reason = "TP"
                    break

        if pnl == 0.0 and actual_idx + 12 < len(val_df):
            exit_idx = actual_idx + 12
            exit_price = val_df['Close'].iloc[exit_idx]
            pnl = (exit_price - price) * lotes if pred == 1 else (price - exit_price) * lotes

        capital += pnl
        equity.append(capital)
        total_trades += 1
        total_lots += lotes
        if exit_reason == "TP":
            tp_hits += 1
        if exit_reason == "TIMEOUT":
            timeout_hits += 1

        if pnl > 0:
            wins += 1
            loss_streak = 0
        else:
            loss_streak += 1
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak

        duration_bars = (exit_idx - actual_idx) if exit_idx is not None else 0
        ts = row.name if hasattr(row, 'name') else datetime.utcnow()
        print(f"{ts}\t{side}\t{price:.2f}\t{exit_price:.2f}\t{pnl:.2f}\t{duration_bars} velas")

        if exit_idx is None:
            esta_en_operacion = False
            i += 1
        else:
            esta_en_operacion = False
            i = exit_idx + 1

    net_profit = capital - config.CAPITAL_INICIAL
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    max_dd = max_drawdown_pct(equity)

    avg_lot = (total_lots / total_trades) if total_trades else 0.0

    print(f"NET PROFIT: {net_profit:.2f}")
    print(f"MAX DRAWDOWN %: {max_dd:.2f}")
    print(f"WIN RATE: {win_rate:.2f}%")
    print(f"RACHA MAX PERDIDAS: {max_loss_streak}")
    print(f"LOTAJE PROMEDIO: {avg_lot:.4f}")
    print(f"TP HITS: {tp_hits}")
    print(f"TIMEOUT HITS: {timeout_hits}")


if __name__ == "__main__":
    main()
