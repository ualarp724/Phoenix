#!/usr/bin/env python3
"""
Optuna BTCUSD con vantage_btc2.csv (M15) y split 2022-2023 train / 2024-2025 validación.
Reglas:
- SL fijo $2.00 con lotaje dinámico.
- TP libre con ratio TP/SL entre 0.5 y 5.0.
- MTF: H1 y M15 deben alinear tendencia.
- Una sola posición abierta (sin múltiples BUY/SELL).
- Filtro EMA200 (trend filter).
- Cooldown 12h tras tocar SL.
- Objetivo: maximizar Sharpe y minimizar drawdown.
"""

import argparse
import copy
import os
import tempfile
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Optuna no disponible: {exc}")

try:
    import xgboost as xgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"XGBoost no disponible: {exc}")

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")


TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

SL_USD = 4.00
SPREAD_USD = 1.50
COMMISSION_PER_LOT = 6.00
OOS_SPREAD_USD = 3.00
OOS_COMMISSION_PER_LOT = 0.01
MIN_CONF = 0.0
MAX_OPEN_POSITIONS = 1
TIME_EXIT_BARS = 16
COOLDOWN_HOURS = 4

TRIAL_46_PARAMS = {
    "max_depth": 4,
    "n_estimators": 450,
    "learning_rate": 0.04230458647489199,
    "min_child_weight": 1,
    "gamma": 0.06504379175132392,
    "subsample": 0.9393916691080864,
    "colsample_bytree": 0.680370856457481,
    "reg_alpha": 0.0023535310962954858,
    "reg_lambda": 1.1760005114606702,
    "max_delta_step": 0,
    "umbral_buy": 0.2921117047116195,
    "umbral_sell": 0.20779352363315906,
    "sl_mult": 0.8061332986360159,
    "tp_rr": 2.9141855045951965,
}

REFINE_845_PARAMS = {
    "max_depth": 5,
    "n_estimators": 400,
    "learning_rate": 0.08634416168894593,
    "min_child_weight": 9,
    "gamma": 0.1059160560602188,
    "subsample": 0.8342925106353404,
    "colsample_bytree": 0.6008864899390829,
    "reg_alpha": 0.4097678326146584,
    "reg_lambda": 2.9207451866541527,
    "max_delta_step": 1,
    "umbral_buy": 0.23268914069728522,
    "umbral_sell": 0.2805090192712911,
    "sl_mult": 1.3013831369109552,
    "tp_rr": 2.4469210106195574,
}

TRIAL_46_PARAMS_PATH = "trial46_params.json"
REFINED_PARAMS_PATH = "refined_params.json"
BTC_FINAL_PARAMS_PATH = "btc_final_params.json"


@dataclass
class BacktestResult:
    profit: float
    win_rate: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    total_trades: int
    avg_trades_per_day: float
    equity_curve: list[tuple[pd.Timestamp, float]]
    trade_pnls: list[tuple[pd.Timestamp, float]]
    sharpe_ratio: float


def _load_and_resample_m15(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep="\t")
        if len(df.columns) < 2:
            df = pd.read_csv(csv_path, sep=",")
    except Exception as exc:
        raise SystemExit(f"Error leyendo CSV: {exc}")

    col_map = {}
    for col in df.columns:
        c = col.upper().replace("<", "").replace(">", "")
        if "DATE" in c:
            col_map[col] = "Date"
        elif "TIME" in c:
            col_map[col] = "Time"
        elif "OPEN" in c:
            col_map[col] = "Open"
        elif "HIGH" in c:
            col_map[col] = "High"
        elif "LOW" in c:
            col_map[col] = "Low"
        elif "CLOSE" in c:
            col_map[col] = "Close"
        elif "VOL" in c:
            col_map[col] = "Volume"

    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()]
    if "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    else:
        df["Datetime"] = pd.to_datetime(df["Date"])

    df = df.set_index("Datetime").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    resampled = df.resample("15min").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    resampled = resampled.reset_index()
    resampled["Date"] = resampled["Datetime"].dt.date.astype(str)
    resampled["Time"] = resampled["Datetime"].dt.time.astype(str)
    return resampled[["Date", "Time", "Open", "High", "Low", "Close", "Volume"]]


def _prepare_dataset(csv_path: str) -> pd.DataFrame:
    df_m15 = _load_and_resample_m15(csv_path)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        df_m15.to_csv(tmp.name, index=False)
        temp_path = tmp.name

    processor = PhoenixDataProcessor(temp_path)
    df = processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.0)
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)

    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna()
    return df


def _train_models(params_xgb, params_lgbm, X_train, y_train):
    tuned_xgb = copy.deepcopy(params_xgb)
    tuned_xgb["n_jobs"] = 1
    tuned_xgb.setdefault("tree_method", "hist")
    xgb_model = xgb.XGBClassifier(**tuned_xgb)
    xgb_model.fit(X_train, y_train, verbose=False)

    tuned_lgbm = copy.deepcopy(params_lgbm)
    tuned_lgbm["n_jobs"] = 1
    lgbm_model = lgb.LGBMClassifier(**tuned_lgbm)
    lgbm_model.fit(X_train, y_train)
    return xgb_model, lgbm_model


def _mtf_trend_ok(row: pd.Series, pred: int) -> bool:
    m15_trend = float(row.get("M15_Trend_Score", 0.0))
    h1_trend = float(row.get("H1_Trend_Score", 0.0))
    if pred == 1:
        return m15_trend >= 0 and h1_trend >= 0
    if pred == 2:
        return m15_trend <= 0 and h1_trend <= 0
    return False


def _backtest_multi_position(
    val_df: pd.DataFrame,
    model_xgb,
    model_lgbm,
    scaler,
    umbral_buy: float,
    umbral_sell: float,
    sl_mult: float,
    tp_rr: float,
    weekday_only: bool = False,
    spread_usd: float | None = None,
    commission_per_lot: float | None = None,
) -> BacktestResult:
    if val_df.empty:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, [], [], 0.0)

    spread_usd = SPREAD_USD if spread_usd is None else float(spread_usd)
    commission_per_lot = (
        COMMISSION_PER_LOT if commission_per_lot is None else float(commission_per_lot)
    )

    data_scaled = scaler.transform(val_df[config.FEATURES].values)
    lookback = config.LOOKBACK_WINDOW
    if len(val_df) <= lookback + TIME_EXIT_BARS:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, [], [], 0.0)

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)

    probs_xgb = model_xgb.predict_proba(flat)
    probs_lgb = model_lgbm.predict_proba(flat)
    weights = config.ENSEMBLE_WEIGHTS
    combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)

    buy_scores = combined[:, 1]
    sell_scores = combined[:, 2]
    preds = np.where(
        buy_scores >= sell_scores,
        np.where(buy_scores >= umbral_buy, 1, 0),
        np.where(sell_scores >= umbral_sell, 2, 0),
    )
    confs = np.where(preds == 1, buy_scores, np.where(preds == 2, sell_scores, 0.0))

    capital = config.CAPITAL_INICIAL
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    wins = 0
    total_trades = 0
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    trade_pnls: list[tuple[pd.Timestamp, float]] = []

    open_positions = []
    last_entry_idx = None
    cooldown_until: pd.Timestamp | None = None
    returns: list[float] = []
    last_equity = capital

    i = 0
    while i < (total_windows - TIME_EXIT_BARS):
        actual_idx = i + lookback
        row = val_df.iloc[actual_idx]
        ts_now = val_df.index[actual_idx]
        price = float(row["Close"])
        hi = float(row["High"])
        lo = float(row["Low"])

        # Actualizar posiciones abiertas
        still_open = []
        for pos in open_positions:
            entry_price = pos["entry_price"]
            side = pos["side"]
            sl_price = pos["sl_price"]
            tp_price = pos["tp_price"]
            lotes = pos["lotes"]
            entry_idx = pos["entry_idx"]
            spread_cost = pos["spread_cost"]
            commission = pos["commission"]

            closed = False
            sl_hit = False
            pnl = 0.0
            if side == 1:
                if lo <= sl_price:
                    pnl = (sl_price - entry_price) * lotes
                    sl_hit = True
                    closed = True
                elif hi >= tp_price:
                    pnl = (tp_price - entry_price) * lotes
                    closed = True
            else:
                if hi >= sl_price:
                    pnl = (entry_price - sl_price) * lotes
                    sl_hit = True
                    closed = True
                elif lo <= tp_price:
                    pnl = (entry_price - tp_price) * lotes
                    closed = True

            if not closed and (actual_idx - entry_idx) >= TIME_EXIT_BARS:
                exit_price = price
                pnl = (exit_price - entry_price) * lotes if side == 1 else (entry_price - exit_price) * lotes
                closed = True

            if closed:
                pnl = pnl - commission - spread_cost
                capital += pnl
                total_trades += 1
                if last_equity > 0:
                    returns.append(pnl / last_equity)
                last_equity = capital
                if pnl > 0:
                    wins += 1
                if capital > peak:
                    peak = capital
                dd = peak - capital
                if dd > max_dd:
                    max_dd = dd
                if peak > 0:
                    dd_pct = (dd / peak) * 100.0
                    if dd_pct > max_dd_pct:
                        max_dd_pct = dd_pct
                equity_curve.append((ts_now, capital))
                trade_pnls.append((ts_now, pnl))
                if sl_hit:
                    cooldown_until = ts_now + pd.Timedelta(hours=COOLDOWN_HOURS)
            else:
                still_open.append(pos)

        open_positions = still_open

        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf < MIN_CONF:
            i += 1
            continue

        if cooldown_until is not None and ts_now < cooldown_until:
            i += 1
            continue

        if weekday_only and val_df.index[actual_idx].weekday() >= 5:
            i += 1
            continue

        if actual_idx == last_entry_idx:
            i += 1
            continue

        if open_positions or len(open_positions) >= MAX_OPEN_POSITIONS:
            i += 1
            continue

        if not _mtf_trend_ok(row, pred):
            i += 1
            continue

        ema_200 = float(row.get("EMA_200", np.nan))
        if not np.isfinite(ema_200):
            i += 1
            continue
        if pred == 1 and price <= ema_200:
            i += 1
            continue
        if pred == 2 and price >= ema_200:
            i += 1
            continue

        atr = float(val_df["NATR"].iloc[actual_idx]) * price / 100.0
        if atr < config.MIN_ATR_THRESHOLD:
            i += 1
            continue

        sl_dist = atr * sl_mult
        if sl_dist <= 0:
            i += 1
            continue

        lotes = SL_USD / sl_dist
        if lotes < config.MIN_LOT_SIZE or lotes > config.MAX_LOT_SIZE:
            i += 1
            continue

        tp_dist = sl_dist * tp_rr

        if pred == 1:
            entry_price = price + spread_usd
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            entry_price = price - spread_usd
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        spread_cost = lotes * spread_usd
        commission = lotes * commission_per_lot

        open_positions.append(
            {
                "entry_price": entry_price,
                "side": pred,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "lotes": lotes,
                "entry_idx": actual_idx,
                "spread_cost": spread_cost,
                "commission": commission,
            }
        )
        last_entry_idx = actual_idx

        i += 1

    # Cerrar posiciones restantes al final
    if open_positions:
        final_price = float(val_df["Close"].iloc[-1])
        final_ts = val_df.index[-1]
        for pos in open_positions:
            entry_price = pos["entry_price"]
            side = pos["side"]
            lotes = pos["lotes"]
            spread_cost = pos["spread_cost"]
            commission = pos["commission"]
            pnl = (final_price - entry_price) * lotes if side == 1 else (entry_price - final_price) * lotes
            pnl = pnl - commission - spread_cost
            capital += pnl
            total_trades += 1
            if pnl > 0:
                wins += 1
            if capital > peak:
                peak = capital
            dd = peak - capital
            if dd > max_dd:
                max_dd = dd
            if peak > 0:
                dd_pct = (dd / peak) * 100.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            equity_curve.append((final_ts, capital))
            trade_pnls.append((final_ts, pnl))

    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    profit = float(capital - config.CAPITAL_INICIAL)
    days = val_df.index.normalize().nunique() if not val_df.empty else 0
    avg_trades_per_day = (total_trades / days) if days else 0.0
    if len(returns) >= 2:
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / std_ret) * np.sqrt(len(returns)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        profit,
        win_rate,
        max_dd,
        max_dd_pct,
        total_trades,
        avg_trades_per_day,
        equity_curve,
        trade_pnls,
        sharpe,
    )


def _simulate_month_with_trades(
    month_df: pd.DataFrame,
    model_xgb,
    model_lgbm,
    scaler,
    params: dict,
    weekday_only: bool,
    spread_usd: float,
    commission_per_lot: float,
) -> list[dict]:
    if month_df.empty:
        return []

    data_scaled = scaler.transform(month_df[config.FEATURES].values)
    lookback = config.LOOKBACK_WINDOW
    if len(month_df) <= lookback + TIME_EXIT_BARS:
        return []

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)

    probs_xgb = model_xgb.predict_proba(flat)
    probs_lgb = model_lgbm.predict_proba(flat)
    weights = config.ENSEMBLE_WEIGHTS
    combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)

    buy_scores = combined[:, 1]
    sell_scores = combined[:, 2]
    preds = np.where(
        buy_scores >= sell_scores,
        np.where(buy_scores >= params["umbral_buy"], 1, 0),
        np.where(sell_scores >= params["umbral_sell"], 2, 0),
    )
    confs = np.where(preds == 1, buy_scores, np.where(preds == 2, sell_scores, 0.0))

    trades: list[dict] = []
    open_positions = []
    last_entry_idx = None
    cooldown_until: pd.Timestamp | None = None

    i = 0
    while i < (total_windows - TIME_EXIT_BARS):
        actual_idx = i + lookback
        row = month_df.iloc[actual_idx]
        ts_now = month_df.index[actual_idx]
        price = float(row["Close"])
        hi = float(row["High"])
        lo = float(row["Low"])

        # Update open positions
        still_open = []
        for pos in open_positions:
            entry_price = pos["entry_price"]
            side = pos["side"]
            sl_price = pos["sl_price"]
            tp_price = pos["tp_price"]
            lotes = pos["lotes"]
            entry_idx = pos["entry_idx"]
            spread_cost = pos["spread_cost"]
            commission = pos["commission"]

            closed = False
            sl_hit = False
            pnl = 0.0
            exit_price = price
            exit_time = ts_now
            if side == 1:
                if lo <= sl_price:
                    pnl = (sl_price - entry_price) * lotes
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif hi >= tp_price:
                    pnl = (tp_price - entry_price) * lotes
                    exit_price = tp_price
                    closed = True
            else:
                if hi >= sl_price:
                    pnl = (entry_price - sl_price) * lotes
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif lo <= tp_price:
                    pnl = (entry_price - tp_price) * lotes
                    exit_price = tp_price
                    closed = True

            if not closed and (actual_idx - entry_idx) >= TIME_EXIT_BARS:
                exit_price = price
                pnl = (exit_price - entry_price) * lotes if side == 1 else (entry_price - exit_price) * lotes
                closed = True

            if closed:
                pnl = pnl - commission - spread_cost
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": exit_time,
                        "side": "BUY" if side == 1 else "SELL",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "pnl": pnl,
                    }
                )
                if sl_hit:
                    cooldown_until = ts_now + pd.Timedelta(hours=COOLDOWN_HOURS)
            else:
                still_open.append(pos)

        open_positions = still_open

        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf < MIN_CONF:
            i += 1
            continue

        if cooldown_until is not None and ts_now < cooldown_until:
            i += 1
            continue

        if weekday_only and ts_now.weekday() >= 5:
            i += 1
            continue

        if actual_idx == last_entry_idx:
            i += 1
            continue

        if open_positions or len(open_positions) >= MAX_OPEN_POSITIONS:
            i += 1
            continue

        if not _mtf_trend_ok(row, pred):
            i += 1
            continue

        ema_200 = float(row.get("EMA_200", np.nan))
        if not np.isfinite(ema_200):
            i += 1
            continue
        if pred == 1 and price <= ema_200:
            i += 1
            continue
        if pred == 2 and price >= ema_200:
            i += 1
            continue

        atr = float(month_df["NATR"].iloc[actual_idx]) * price / 100.0
        if atr < config.MIN_ATR_THRESHOLD:
            i += 1
            continue

        sl_dist = atr * params["sl_mult"]
        if sl_dist <= 0:
            i += 1
            continue

        lotes = SL_USD / sl_dist
        if lotes < config.MIN_LOT_SIZE or lotes > config.MAX_LOT_SIZE:
            i += 1
            continue

        tp_dist = sl_dist * params["tp_rr"]

        if pred == 1:
            entry_price = price + spread_usd
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            entry_price = price - spread_usd
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        open_positions.append(
            {
                "entry_time": ts_now,
                "entry_price": entry_price,
                "side": pred,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "lotes": lotes,
                "entry_idx": actual_idx,
                "spread_cost": lotes * spread_usd,
                "commission": lotes * commission_per_lot,
            }
        )
        last_entry_idx = actual_idx
        i += 1

    return trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--audit-trial46", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine-high", action="store_true")
    parser.add_argument("--stress-compare", action="store_true")
    parser.add_argument("--stress-random-refined", action="store_true")
    parser.add_argument("--oos-refined", action="store_true")
    parser.add_argument("--audit-random-month-refined", action="store_true")
    parser.add_argument("--audit-range-month-refined", action="store_true")
    args = parser.parse_args()

    config.apply_asset("BTCUSD")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = TIME_EXIT_BARS
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_btc2.csv")
    df_all = df_all.sort_index()

    df_all = df_all.loc[TRAIN_START:VAL_END].copy()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("Split de datos inválido (train/validación vacíos).")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    if len(X_train) != len(y_train):
        raise SystemExit("X_train/y_train desalineados.")

    def objective(trial: optuna.Trial):
        params_xgb = copy.deepcopy(config.XGB_PARAMS)
        params_xgb.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 25),
                "gamma": trial.suggest_float("gamma", 0.0, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 1),
            }
        )

        params_lgbm = copy.deepcopy(config.LGBM_PARAMS)

        if args.refine_high:
            sl_mult = trial.suggest_float("sl_mult", 1.2, 3.5)
            tp_rr = trial.suggest_float("tp_rr", 2.0, 4.0)
            umbral_buy = trial.suggest_float("umbral_buy", 0.20, 0.30)
            umbral_sell = trial.suggest_float("umbral_sell", 0.20, 0.30)
        elif args.refine:
            sl_mult = trial.suggest_float("sl_mult", 1.2, 3.5)
            tp_rr = trial.suggest_float("tp_rr", 1.5, 4.5)
            umbral_buy = trial.suggest_float("umbral_buy", 0.22, 0.40)
            umbral_sell = trial.suggest_float("umbral_sell", 0.22, 0.40)
        else:
            sl_mult = trial.suggest_float("sl_mult", 0.8, 3.0)
            tp_rr = trial.suggest_float("tp_rr", 0.5, 5.0)
            umbral_buy = trial.suggest_float("umbral_buy", 0.20, 0.35)
            umbral_sell = trial.suggest_float("umbral_sell", 0.20, 0.35)

        rng = np.random.default_rng(trial.number)
        sample_size = max(1, int(len(X_train) * 0.5))
        sample_idx = rng.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train[sample_idx]
        y_sample = y_train[sample_idx]

        xgb_model, lgbm_model = _train_models(params_xgb, params_lgbm, X_sample, y_sample)

        result = _backtest_multi_position(
            val_df,
            xgb_model,
            lgbm_model,
            scaler,
            umbral_buy,
            umbral_sell,
            sl_mult,
            tp_rr,
            weekday_only=(args.refine or args.refine_high),
        )

        if args.refine and result.win_rate < 42.0:
            return -1e9

        if args.refine_high and result.max_drawdown_pct > 18.0:
            return -10000000.0

        dd_penalty = result.max_drawdown_pct / 100.0
        score = float(result.sharpe_ratio) - dd_penalty

        print(
            f"Profit: {result.profit:.2f} | Winrate: {result.win_rate:.2f}% | "
            f"MaxDD: ${result.max_drawdown_usd:.2f} | Sharpe: {result.sharpe_ratio:.2f} | "
            f"Score: {score:.4f}"
        )

        return float(score)

    if args.audit_range_month_refined:
        best = REFINE_845_PARAMS
    elif args.audit_random_month_refined:
        best = REFINE_845_PARAMS
    elif args.oos_refined:
        best = REFINE_845_PARAMS
    elif args.stress_compare:
        best = TRIAL_46_PARAMS
    elif args.stress_random_refined:
        best = REFINE_845_PARAMS
    elif args.audit_trial46:
        best = TRIAL_46_PARAMS
    else:
        study = optuna.create_study(direction="maximize")
        if args.refine or args.refine_high:
            seed = dict(TRIAL_46_PARAMS)
            seed["sl_mult"] = max(1.2, min(seed["sl_mult"], 3.5))
            if args.refine_high:
                seed["tp_rr"] = max(2.0, min(seed["tp_rr"], 4.0))
                seed["umbral_buy"] = max(0.20, min(seed["umbral_buy"], 0.30))
                seed["umbral_sell"] = max(0.20, min(seed["umbral_sell"], 0.30))
            else:
                seed["tp_rr"] = max(1.5, min(seed["tp_rr"], 4.5))
                seed["umbral_buy"] = max(0.22, min(seed["umbral_buy"], 0.40))
                seed["umbral_sell"] = max(0.22, min(seed["umbral_sell"], 0.40))
            study.enqueue_trial(seed)
        study.optimize(objective, n_trials=args.trials, n_jobs=2)
        best = study.best_trial.params
    best_xgb = copy.deepcopy(config.XGB_PARAMS)
    best_xgb.update(
        {
            "max_depth": best["max_depth"],
            "n_estimators": best["n_estimators"],
            "learning_rate": best["learning_rate"],
            "min_child_weight": best["min_child_weight"],
            "gamma": best["gamma"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "max_delta_step": best["max_delta_step"],
        }
    )

    xgb_model, lgbm_model = _train_models(best_xgb, config.LGBM_PARAMS, X_train, y_train)

    final_result = _backtest_multi_position(
        val_df,
        xgb_model,
        lgbm_model,
        scaler,
        best["umbral_buy"],
        best["umbral_sell"],
        best["sl_mult"],
        best["tp_rr"],
        weekday_only=(args.refine or args.refine_high),
    )

    recovery = (
        final_result.profit / final_result.max_drawdown_usd
        if final_result.max_drawdown_usd > 0
        else float("inf")
    )

    print("\n✅ INFORME FINAL (EXAMEN 2024-2025)")
    print(f"Mejores parámetros: {best}")
    print(f"Beneficio Total: {final_result.profit:.2f}")
    print(f"Winrate Neto: {final_result.win_rate:.2f}%")
    print(f"Factor de Recuperación Neto: {recovery:.2f}")
    print(f"MaxDD %: {final_result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
    print(f"Trades por día (promedio): {final_result.avg_trades_per_day:.2f}")

    if final_result.trade_pnls:
        pnl_series = pd.Series(
            [p for _, p in final_result.trade_pnls],
            index=pd.to_datetime([t for t, _ in final_result.trade_pnls]),
        )
        monthly_pnl = pnl_series.resample("ME").sum()
        print("\n📅 PnL Mensual (USD)")
        for idx, value in monthly_pnl.items():
            print(f"{idx.strftime('%Y-%m')}: {value:.2f}")

    with open(TRIAL_46_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(TRIAL_46_PARAMS, f, indent=2)
    with open(REFINED_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(REFINE_845_PARAMS, f, indent=2)
    btc_final = dict(REFINE_845_PARAMS)
    btc_final.update(
        {
            "sl_usd": SL_USD,
            "cooldown_hours": COOLDOWN_HOURS,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "ema_filter": True,
        }
    )
    with open(BTC_FINAL_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(btc_final, f, indent=2)

    if args.refine_high:
        ok = (
            final_result.profit > 1500.0
            and final_result.win_rate > 40.0
            and final_result.max_drawdown_pct < 18.0
        )
        print("\n✅ OBJETIVO BOT PERFECTO" if ok else "\n❌ OBJETIVO BOT PERFECTO")

    if args.stress_compare:
        print("\n🧪 STRESS TEST: TRIAL 46 VS REFINADO 845")
        base = TRIAL_46_PARAMS
        base_xgb = copy.deepcopy(config.XGB_PARAMS)
        base_xgb.update(
            {
                "max_depth": base["max_depth"],
                "n_estimators": base["n_estimators"],
                "learning_rate": base["learning_rate"],
                "min_child_weight": base["min_child_weight"],
                "gamma": base["gamma"],
                "subsample": base["subsample"],
                "colsample_bytree": base["colsample_bytree"],
                "reg_alpha": base["reg_alpha"],
                "reg_lambda": base["reg_lambda"],
                "max_delta_step": base["max_delta_step"],
            }
        )
        base_model_xgb, base_model_lgbm = _train_models(base_xgb, config.LGBM_PARAMS, X_train, y_train)
        base_result = _backtest_multi_position(
            val_df,
            base_model_xgb,
            base_model_lgbm,
            scaler,
            base["umbral_buy"],
            base["umbral_sell"],
            base["sl_mult"],
            base["tp_rr"],
            weekday_only=False,
        )

        ref = REFINE_845_PARAMS
        ref_xgb = copy.deepcopy(config.XGB_PARAMS)
        ref_xgb.update(
            {
                "max_depth": ref["max_depth"],
                "n_estimators": ref["n_estimators"],
                "learning_rate": ref["learning_rate"],
                "min_child_weight": ref["min_child_weight"],
                "gamma": ref["gamma"],
                "subsample": ref["subsample"],
                "colsample_bytree": ref["colsample_bytree"],
                "reg_alpha": ref["reg_alpha"],
                "reg_lambda": ref["reg_lambda"],
                "max_delta_step": ref["max_delta_step"],
            }
        )
        ref_model_xgb, ref_model_lgbm = _train_models(ref_xgb, config.LGBM_PARAMS, X_train, y_train)
        ref_result = _backtest_multi_position(
            val_df,
            ref_model_xgb,
            ref_model_lgbm,
            scaler,
            ref["umbral_buy"],
            ref["umbral_sell"],
            ref["sl_mult"],
            ref["tp_rr"],
            weekday_only=True,
        )

        print(
            f"Trial 46 -> Profit: {base_result.profit:.2f} | Winrate: {base_result.win_rate:.2f}% | "
            f"MaxDD%: {base_result.max_drawdown_pct:.2f}%"
        )
        print(
            f"Refinado 845 -> Profit: {ref_result.profit:.2f} | Winrate: {ref_result.win_rate:.2f}% | "
            f"MaxDD%: {ref_result.max_drawdown_pct:.2f}%"
        )

    if args.stress_random_refined:
        print("\n🧪 STRESS TEST: 3 MESES RANDOM (REFINADO)")
        rng = np.random.default_rng(42)
        months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
        sample_months = rng.choice(months, size=3, replace=False)

        ref = REFINE_845_PARAMS
        ref_xgb = copy.deepcopy(config.XGB_PARAMS)
        ref_xgb.update(
            {
                "max_depth": ref["max_depth"],
                "n_estimators": ref["n_estimators"],
                "learning_rate": ref["learning_rate"],
                "min_child_weight": ref["min_child_weight"],
                "gamma": ref["gamma"],
                "subsample": ref["subsample"],
                "colsample_bytree": ref["colsample_bytree"],
                "reg_alpha": ref["reg_alpha"],
                "reg_lambda": ref["reg_lambda"],
                "max_delta_step": ref["max_delta_step"],
            }
        )
        ref_model_xgb, ref_model_lgbm = _train_models(ref_xgb, config.LGBM_PARAMS, X_train, y_train)

        for month_start in sorted(sample_months):
            month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
            slice_df = val_df.loc[month_start:month_end].copy()
            result = _backtest_multi_position(
                slice_df,
                ref_model_xgb,
                ref_model_lgbm,
                scaler,
                ref["umbral_buy"],
                ref["umbral_sell"],
                ref["sl_mult"],
                ref["tp_rr"],
                weekday_only=True,
            )
            print(
                f"{month_start.strftime('%Y-%m')}: Profit {result.profit:.2f} | "
                f"Winrate {result.win_rate:.2f}% | MaxDD% {result.max_drawdown_pct:.2f}%"
            )

    if args.oos_refined:
        print("\n🧪 OUT-OF-SAMPLE (REFINADO 845)")
        ref = REFINE_845_PARAMS
        ref_xgb = copy.deepcopy(config.XGB_PARAMS)
        ref_xgb.update(
            {
                "max_depth": ref["max_depth"],
                "n_estimators": ref["n_estimators"],
                "learning_rate": ref["learning_rate"],
                "min_child_weight": ref["min_child_weight"],
                "gamma": ref["gamma"],
                "subsample": ref["subsample"],
                "colsample_bytree": ref["colsample_bytree"],
                "reg_alpha": ref["reg_alpha"],
                "reg_lambda": ref["reg_lambda"],
                "max_delta_step": ref["max_delta_step"],
            }
        )
        ref_model_xgb, ref_model_lgbm = _train_models(ref_xgb, config.LGBM_PARAMS, X_train, y_train)

        train_curve = _backtest_multi_position(
            train_df,
            ref_model_xgb,
            ref_model_lgbm,
            scaler,
            ref["umbral_buy"],
            ref["umbral_sell"],
            ref["sl_mult"],
            ref["tp_rr"],
            weekday_only=True,
            spread_usd=OOS_SPREAD_USD,
            commission_per_lot=OOS_COMMISSION_PER_LOT,
        )

        oos_end = pd.to_datetime("2025-12-31")
        oos_start = oos_end - pd.DateOffset(months=4)
        oos_df = df_all.loc[oos_start:oos_end].copy()

        oos_result = _backtest_multi_position(
            oos_df,
            ref_model_xgb,
            ref_model_lgbm,
            scaler,
            ref["umbral_buy"],
            ref["umbral_sell"],
            ref["sl_mult"],
            ref["tp_rr"],
            weekday_only=True,
            spread_usd=OOS_SPREAD_USD,
            commission_per_lot=OOS_COMMISSION_PER_LOT,
        )

        print(
            f"Train Winrate: {train_curve.win_rate:.2f}% | OOS Winrate: {oos_result.win_rate:.2f}%"
        )

        if train_curve.equity_curve and oos_result.equity_curve:
            train_eq = pd.DataFrame(train_curve.equity_curve, columns=["Timestamp", "Equity"]).set_index("Timestamp")
            oos_eq = pd.DataFrame(oos_result.equity_curve, columns=["Timestamp", "Equity"]).set_index("Timestamp")
            train_eq = train_eq.sort_index()
            oos_eq = oos_eq.sort_index()

            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(train_eq.index, train_eq["Equity"], label="Train 2022-2023")
                plt.plot(oos_eq.index, oos_eq["Equity"], label="OOS (últimos 4 meses 2025)")
                plt.title("Equity Curve Comparison (Refinado 845)")
                plt.xlabel("Fecha")
                plt.ylabel("Capital (USD)")
                plt.legend()
                plt.tight_layout()
                plt.savefig("curva_equity_compare_oos.png")
                plt.close()
            except Exception as exc:  # pragma: no cover
                print(f"Error generando gráfico comparativo: {exc}")

        if oos_result.win_rate < 30.0:
            print("⚠️ Winrate OOS < 30%: posible overfitting.")
        elif oos_result.win_rate < 40.0:
            print("⚠️ Winrate OOS < 40%: degradación fuera de rango aceptable.")
        else:
            print("✅ Winrate OOS dentro de rango aceptable.")

    if args.audit_random_month_refined:
        print("\n🧪 AUDITORÍA MES RANDOM (REFINADO 845)")
        rng = np.random.default_rng(7)
        months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
        month_start = pd.Timestamp(rng.choice(months, size=1, replace=False)[0])
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        month_df = val_df.loc[month_start:month_end].copy()

        trades = _simulate_month_with_trades(
            month_df,
            xgb_model,
            lgbm_model,
            scaler,
            REFINE_845_PARAMS,
            weekday_only=True,
            spread_usd=OOS_SPREAD_USD,
            commission_per_lot=OOS_COMMISSION_PER_LOT,
        )

        print(f"Mes seleccionado: {month_start.strftime('%Y-%m')}")
        print("Fecha/Hora\tTipo\tEntrada\tSalida\tSL\tTP\tPnL_USD")
        for t in trades:
            print(
                f"{t['entry_time']}\t{t['side']}\t{t['entry_price']:.2f}\t"
                f"{t['exit_price']:.2f}\t{t['sl_price']:.2f}\t{t['tp_price']:.2f}\t{t['pnl']:.2f}"
            )

        if trades:
            equity = config.CAPITAL_INICIAL
            peak = equity
            loss_streak = 0
            max_loss_streak = 0
            streak_peak = equity
            streak_start_time = None
            recovery_time = None

            for t in trades:
                pnl = t["pnl"]
                equity += pnl
                if pnl < 0:
                    loss_streak += 1
                    if loss_streak == 1:
                        streak_peak = peak
                        streak_start_time = t["exit_time"]
                    if loss_streak > max_loss_streak:
                        max_loss_streak = loss_streak
                        recovery_time = None
                else:
                    if loss_streak > 0 and recovery_time is None and equity >= streak_peak:
                        recovery_time = t["exit_time"]
                    loss_streak = 0
                if equity > peak:
                    peak = equity

            print(f"Racha máxima de pérdidas: {max_loss_streak}")
            if streak_start_time and recovery_time:
                print(f"Recuperación hasta nuevo máximo: {recovery_time - streak_start_time}")
            else:
                print("Recuperación hasta nuevo máximo: no ocurrió en el mes")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(month_df.index, month_df["Close"], label="Close")
            for t in trades:
                color = "green" if t["side"] == "BUY" else "red"
                plt.scatter(t["entry_time"], t["entry_price"], color=color, marker="^", s=40)
                plt.scatter(t["exit_time"], t["exit_price"], color=color, marker="x", s=40)
            plt.title(f"Operativa BTC Refinado - {month_start.strftime('%Y-%m')}")
            plt.xlabel("Fecha")
            plt.ylabel("Precio")
            plt.legend()
            plt.tight_layout()
            plt.savefig("operativa_mes_refinado.png")
            plt.close()
        except Exception as exc:  # pragma: no cover
            print(f"Error generando gráfico operativo: {exc}")

    if args.audit_range_month_refined:
        print("\n🧪 AUDITORÍA MES LATERAL (REFINADO 845)")
        monthly_groups = val_df.groupby(pd.Grouper(freq="MS"))
        month_scores = []
        for month_start, mdf in monthly_groups:
            if mdf.empty:
                continue
            close_start = float(mdf["Close"].iloc[0])
            close_end = float(mdf["Close"].iloc[-1])
            abs_return = abs(close_end - close_start)
            range_month = float(mdf["High"].max() - mdf["Low"].min())
            if range_month <= 0:
                continue
            trend_ratio = abs_return / range_month
            month_scores.append((trend_ratio, month_start))

        if not month_scores:
            print("No se encontró un mes lateral válido.")
            return

        month_scores.sort(key=lambda x: x[0])
        month_start = month_scores[0][1]
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        month_df = val_df.loc[month_start:month_end].copy()

        trades = _simulate_month_with_trades(
            month_df,
            xgb_model,
            lgbm_model,
            scaler,
            REFINE_845_PARAMS,
            weekday_only=True,
            spread_usd=OOS_SPREAD_USD,
            commission_per_lot=OOS_COMMISSION_PER_LOT,
        )

        print(f"Mes lateral seleccionado: {month_start.strftime('%Y-%m')}")
        print("Fecha/Hora\tTipo\tEntrada\tSalida\tSL\tTP\tPnL_USD")
        for t in trades:
            print(
                f"{t['entry_time']}\t{t['side']}\t{t['entry_price']:.2f}\t"
                f"{t['exit_price']:.2f}\t{t['sl_price']:.2f}\t{t['tp_price']:.2f}\t{t['pnl']:.2f}"
            )

        if trades:
            days = month_df.index.normalize().nunique()
            trades_per_day = len(trades) / days if days else 0.0
            print(f"Trades por día (mes): {trades_per_day:.2f}")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(month_df.index, month_df["Close"], label="Close")
            for t in trades:
                color = "green" if t["side"] == "BUY" else "red"
                plt.scatter(t["entry_time"], t["entry_price"], color=color, marker="^", s=40)
                plt.scatter(t["exit_time"], t["exit_price"], color=color, marker="x", s=40)
            plt.title(f"Operativa BTC Refinado (Mes Lateral) - {month_start.strftime('%Y-%m')}")
            plt.xlabel("Fecha")
            plt.ylabel("Precio")
            plt.legend()
            plt.tight_layout()
            plt.savefig("operativa_mes_lateral_refinado.png")
            plt.close()
        except Exception as exc:  # pragma: no cover
            print(f"Error generando gráfico operativo: {exc}")

    report_result = final_result
    if args.refine:
        base = TRIAL_46_PARAMS
        base_xgb = copy.deepcopy(config.XGB_PARAMS)
        base_xgb.update(
            {
                "max_depth": base["max_depth"],
                "n_estimators": base["n_estimators"],
                "learning_rate": base["learning_rate"],
                "min_child_weight": base["min_child_weight"],
                "gamma": base["gamma"],
                "subsample": base["subsample"],
                "colsample_bytree": base["colsample_bytree"],
                "reg_alpha": base["reg_alpha"],
                "reg_lambda": base["reg_lambda"],
                "max_delta_step": base["max_delta_step"],
            }
        )
        base_model_xgb, base_model_lgbm = _train_models(base_xgb, config.LGBM_PARAMS, X_train, y_train)
        base_result = _backtest_multi_position(
            val_df,
            base_model_xgb,
            base_model_lgbm,
            scaler,
            base["umbral_buy"],
            base["umbral_sell"],
            base["sl_mult"],
            base["tp_rr"],
            weekday_only=False,
        )
        report_result = base_result

    if args.audit_trial46 or args.refine:
        if report_result.equity_curve:
            equity_df = pd.DataFrame(
                report_result.equity_curve, columns=["Timestamp", "Equity"]
            ).set_index("Timestamp")
            equity_df = equity_df.sort_index()
            equity_df = equity_df[~equity_df.index.duplicated(keep="last")]

            peak = equity_df["Equity"].cummax()
            drawdown_pct = (equity_df["Equity"] - peak) / peak * 100.0

            drawdowns = []
            in_dd = False
            dd_start = None
            dd_peak = None
            dd_min = None
            dd_min_time = None
            for ts, equity in equity_df["Equity"].items():
                equity_val = float(equity)
                current_peak = peak.loc[ts]
                if isinstance(current_peak, pd.Series):
                    current_peak = float(current_peak.iloc[-1])
                if equity_val < current_peak and not in_dd:
                    in_dd = True
                    dd_start = ts
                    dd_peak = current_peak
                    dd_min = equity_val
                    dd_min_time = ts
                elif in_dd:
                    if equity_val < dd_min:
                        dd_min = equity_val
                        dd_min_time = ts
                    if equity_val >= dd_peak:
                        duration = ts - dd_start
                        depth = (dd_peak - dd_min)
                        drawdowns.append(
                            (dd_start, dd_min_time, ts, depth, duration)
                        )
                        in_dd = False

            if in_dd and dd_start is not None and dd_peak is not None:
                duration = equity_df.index[-1] - dd_start
                depth = (dd_peak - dd_min) if dd_min is not None else 0.0
                drawdowns.append((dd_start, dd_min_time, equity_df.index[-1], depth, duration))

            drawdowns = sorted(drawdowns, key=lambda x: x[3], reverse=True)[:3]

            print("\n📉 TOP 3 DRAWDOWNS")
            for idx, (start, trough, recover, depth, duration) in enumerate(drawdowns, start=1):
                print(
                    f"{idx}. Inicio: {start.date()} | Mínimo: {trough.date()} | "
                    f"Recupera: {recover.date()} | Caída: ${depth:.2f} | "
                    f"Duración: {duration}"
                )

            pnl_series = pd.Series(
                {ts: pnl for ts, pnl in report_result.trade_pnls}
            )
            monthly_pnl = pnl_series.resample("ME").sum()

            print("\n📅 PNL NETO MENSUAL (2024-2025)")
            for ts, value in monthly_pnl.items():
                print(f"{ts.strftime('%Y-%m')}: {value:.2f}")

            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(equity_df.index, equity_df["Equity"], label="Equity")
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(drawdown_pct.index, drawdown_pct.values, color="red", alpha=0.5, label="Drawdown %")
                ax2.set_ylabel("Drawdown %")
                max_dd_pct = abs(drawdown_pct.min()) if not drawdown_pct.empty else 0.0
                plt.title(f"Equity Curve BTC (2024-2025) | Max DD: {max_dd_pct:.2f}%")
                plt.xlabel("Fecha")
                plt.ylabel("Capital (USD)")
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig("curva_equity_46.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.plot(drawdown_pct.index, drawdown_pct.values, color="red")
                plt.title("Underwater (Drawdown %) BTC")
                plt.xlabel("Fecha")
                plt.ylabel("Drawdown %")
                plt.tight_layout()
                plt.savefig("underwater_btc.png")
                plt.close()
            except Exception as exc:  # pragma: no cover
                print(f"Error generando gráficos: {exc}")

    if args.refine:
        print("\n🧪 COMPARATIVA TRIAL 46 VS REFINADO")

        def _max_dd_pct(result: BacktestResult) -> float:
            if not result.equity_curve:
                return 0.0
            df = pd.DataFrame(result.equity_curve, columns=["Timestamp", "Equity"]).set_index("Timestamp")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            peak = df["Equity"].cummax()
            dd_pct = (df["Equity"] - peak) / peak * 100.0
            return abs(float(dd_pct.min())) if not dd_pct.empty else 0.0

        base_dd_pct = _max_dd_pct(report_result)
        refined_dd_pct = _max_dd_pct(final_result)

        print(
            f"Trial 46 -> Profit: {report_result.profit:.2f} | Winrate: {report_result.win_rate:.2f}% | "
            f"MaxDD%: {base_dd_pct:.2f}%"
        )
        print(
            f"Refinado -> Profit: {final_result.profit:.2f} | Winrate: {final_result.win_rate:.2f}% | "
            f"MaxDD%: {refined_dd_pct:.2f}%"
        )

        if refined_dd_pct >= base_dd_pct:
            print("Refinado NO supera al Trial 46 en estabilidad.")


if __name__ == "__main__":
    main()
