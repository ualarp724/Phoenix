#!/usr/bin/env python3
"""
Optuna XAUUSD con vantage_gold.csv (M15) y split 2022-2023 train / 2024-2025 validación.
Reglas:
- Filtro EMA200: solo largos si el precio está por encima, cortos si está por debajo.
- SL dinámico vía ATR (adaptativo a volatilidad).
- TP/RR entre 2.3 y 3.0 (baseline M5 extrapolado a M15).
- MTF: H1 y M15 deben alinear tendencia.
"""

import argparse
import copy
import tempfile
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

RISK_PER_TRADE_USD = 5.00
SPREAD_POINTS = 1.0
COMMISSION_PER_LOT = 0.0
MIN_CONF = 0.0
MAX_OPEN_POSITIONS = 5
TIME_EXIT_BARS = 16
EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_BUY_MAX = 70
RSI_SELL_MIN = 30
AVOID_NY_OPEN = False
NY_OPEN_START = 13
NY_OPEN_END = 16


@dataclass
class BacktestResult:
    profit: float
    win_rate: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    total_trades: int
    avg_trades_per_day: float
    sharpe: float
    equity_curve: list[tuple[pd.Timestamp, float]]
    trade_pnls: list[tuple[pd.Timestamp, float]]


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

    if "EMA200" not in df.columns and "Close" in df.columns:
        df["EMA200"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()

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
        # Placeholder: lógica de tendencia MTF BUY
        return m15_trend > 0.2 and h1_trend > 0.2

    data_scaled = scaler.transform(val_df[config.FEATURES].values)
    val_df = val_df.copy()
    val_df["RSI"] = _rsi(val_df["Close"]).clip(0, 100)
    lookback = config.LOOKBACK_WINDOW
    if len(val_df) <= lookback + TIME_EXIT_BARS:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, [], [])

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

    capital = config.CAPITAL_INICIAL
    peak = capital
    max_dd = 0.0
    trades = 0
    wins = 0
    equity_curve = []
    trade_pnls = []
    open_positions = []

    for i in range(lookback, len(val_df) - TIME_EXIT_BARS):
        row = val_df.iloc[i]
        ts = val_df.index[i]
        pred = int(preds[i - lookback])

        if weekday_only and ts.weekday() > 4:
            continue

        if AVOID_NY_OPEN and NY_OPEN_START <= ts.hour <= NY_OPEN_END:
            continue

        if pred == 0:
            continue

        if not _mtf_trend_ok(row, pred):
            continue

        if not _ema_filter_ok(row, pred):
            continue

        rsi_val = float(row.get("RSI", 50.0))
        if pred == 1 and rsi_val > RSI_BUY_MAX:
            continue
        if pred == 2 and rsi_val < RSI_SELL_MIN:
            continue

        atr = float(row.get("NATR", 0.0)) * float(row.get("Close", 0.0)) / 100.0
        if atr <= 0:
            continue

        sl_dist = atr * sl_mult
        tp_dist = sl_dist * tp_rr

        entry = float(row.get("Close", 0.0))
        if pred == 1:
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        risk_usd = RISK_PER_TRADE_USD
        lot_size = max(risk_usd / max(sl_dist, 1e-6), 0.01)
        open_positions.append(
            {
                "direction": pred,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "lot": lot_size,
                "entry_idx": i,
            }
        )

        to_close = []
        for idx, pos in enumerate(open_positions):
            if i <= pos["entry_idx"]:
                continue

            hi = float(val_df.iloc[i]["High"])
            lo = float(val_df.iloc[i]["Low"])
            pnl = None

            if pos["direction"] == 1:
                if lo <= pos["sl"]:
                    pnl = -abs(pos["entry"] - pos["sl"]) * pos["lot"]
                elif hi >= pos["tp"]:
                    pnl = abs(pos["tp"] - pos["entry"]) * pos["lot"]
            else:
                if hi >= pos["sl"]:
                    pnl = -abs(pos["sl"] - pos["entry"]) * pos["lot"]
                elif lo <= pos["tp"]:
                    pnl = abs(pos["entry"] - pos["tp"]) * pos["lot"]

            if pnl is None and (i - pos["entry_idx"]) >= TIME_EXIT_BARS:
                exit_price = float(val_df.iloc[i]["Close"])
                pnl = (
                    (exit_price - pos["entry"]) * pos["lot"]
                    if pos["direction"] == 1
                    else (pos["entry"] - exit_price) * pos["lot"]
                )

            if pnl is not None:
                trades += 1
                if pnl > 0:
                    wins += 1
                capital += pnl
                equity_curve.append((ts, capital))
                trade_pnls.append((ts, pnl))
                peak = max(peak, capital)
                max_dd = max(max_dd, (peak - capital))
                to_close.append(idx)

        for idx in sorted(to_close, reverse=True):
            open_positions.pop(idx)

    win_rate = (wins / trades * 100) if trades else 0.0
    max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0.0
    avg_trades_per_day = trades / max((val_df.index[-1] - val_df.index[0]).days, 1)
    sharpe = _compute_sharpe(trade_pnls, config.CAPITAL_INICIAL)

    return BacktestResult(
        profit=capital - config.CAPITAL_INICIAL,
        win_rate=win_rate,
        max_drawdown_usd=max_dd,
        max_drawdown_pct=max_dd_pct,
        total_trades=trades,
        avg_trades_per_day=avg_trades_per_day,
        sharpe=sharpe,
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    config.apply_asset("XAUUSD")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = TIME_EXIT_BARS
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_gold.csv")
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

    first_sharpe_hit = {"printed": False}

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

        # Extrapolación desde M5 (umbral ~0.19, SL~1.48 ATR, TP~1.56 ATR)
        sl_mult = trial.suggest_float("sl_mult", 0.9, 1.8)
        tp_rr = trial.suggest_float("tp_rr", 2.3, 3.0)
        umbral_buy = trial.suggest_float("umbral_buy", 0.10, 0.24)
        umbral_sell = trial.suggest_float("umbral_sell", 0.10, 0.24)

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
            weekday_only=False,
        )

        # No hard filter on trades/day for XAUUSD; allow learning from low-activity regimes.

        if result.sharpe >= 1.5 and not first_sharpe_hit["printed"]:
            first_sharpe_hit["printed"] = True
            print(
                "\n✅ PRIMER TRIAL CON SHARPE >= 1.5"
                f"\nTrial: {trial.number} | Sharpe: {result.sharpe:.2f} | Profit: {result.profit:.2f} | "
                f"Winrate: {result.win_rate:.2f}% | Trades: {result.total_trades}"
            )

        score = float(result.profit)

        print(
            f"Profit: {result.profit:.2f} | Winrate: {result.win_rate:.2f}% | "
            f"Sharpe: {result.sharpe:.2f} | Trades/día: {result.avg_trades_per_day:.2f} | "
            f"Score: {score:.4f}"
        )

        return float(score)

    study = optuna.create_study(direction="maximize")
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
        weekday_only=False,
    )

    print("\n✅ INFORME FINAL (XAUUSD 2024-2025)")
    print(f"Beneficio Total: {final_result.profit:.2f}")
    print(f"Winrate Neto: {final_result.win_rate:.2f}%")
    print(f"Sharpe: {final_result.sharpe:.2f}")
    print(f"Trades por día (promedio): {final_result.avg_trades_per_day:.2f}")


if __name__ == "__main__":
    main()
