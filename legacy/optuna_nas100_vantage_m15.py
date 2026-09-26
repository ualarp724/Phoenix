#!/usr/bin/env python3
"""
Optuna NAS100 con vantage_nas100.csv (M15) y split 2024 train / 2025 validación.
Reglas:
- Filtro EMA200: solo largos si el precio está por encima, cortos si está por debajo.
- SL dinámico vía ATR (adaptativo a volatilidad).
- TP/RR entre 2.0 y 3.0 para capturar impulsos.
- MTF: H1 y M15 deben alinear tendencia.
- Objetivo: maximizar profit con Sharpe > 1.5 y frecuencia >= 1 trade/día.
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

RISK_PER_TRADE_USD = 2.00
SPREAD_POINTS = 1.0
COMMISSION_PER_LOT = 0.0
MIN_CONF = 0.0
MAX_OPEN_POSITIONS = 3
TIME_EXIT_BARS = 16
EMA_PERIOD = 200


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
    # --- OBJETIVE EXTERNO PARA OPTIMIZACIÓN MULTI-ASSET ---
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
            commission = pos["commission"]

            closed = False
            pnl = 0.0
            if side == 1:
                if lo <= sl_price:
                    pnl = (sl_price - entry_price) * lotes
                    closed = True
                elif hi >= tp_price:
                    pnl = (tp_price - entry_price) * lotes
                    closed = True
            else:
                if hi >= sl_price:
                    pnl = (entry_price - sl_price) * lotes
def objective(trial, custom_params=None):
    config.apply_asset("NAS100")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = TIME_EXIT_BARS
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_nas100.csv")
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

    # Hiperparámetros
    if custom_params:
        sl_mult = custom_params.get('sl_mult', 1.0)
        tp_rr = custom_params.get('tp_rr', 2.0)
        umbral_buy = custom_params.get('umbral_buy', 0.15)
        umbral_sell = custom_params.get('umbral_sell', 0.15)
    else:
        sl_mult = trial.suggest_float('sl_mult', 0.8, 3.0)
        tp_rr = trial.suggest_float('tp_rr', 2.0, 3.0)
        umbral_buy = trial.suggest_float('umbral_buy', 0.10, 0.30)
        umbral_sell = trial.suggest_float('umbral_sell', 0.10, 0.30)

    # Modelos (usar los defaults, o parametrizar si se desea)
    params_xgb = copy.deepcopy(config.XGB_PARAMS)
    params_lgbm = copy.deepcopy(config.LGBM_PARAMS)
    xgb_model, lgbm_model = _train_models(params_xgb, params_lgbm, X_train, y_train)

    # Backtest
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
    # Cerrar posiciones restantes al final
    if open_positions:
        final_price = float(val_df["Close"].iloc[-1])
        final_ts = val_df.index[-1]
        for pos in open_positions:
            entry_price = pos["entry_price"]
            side = pos["side"]
            lotes = pos["lotes"]
            commission = pos["commission"]
            pnl = (final_price - entry_price) * lotes if side == 1 else (entry_price - final_price) * lotes
            pnl = pnl - commission
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
    sharpe = _compute_sharpe(trade_pnls, config.CAPITAL_INICIAL)

    return BacktestResult(
        profit,
        win_rate,
        max_dd,
        max_dd_pct,
        total_trades,
        avg_trades_per_day,
        sharpe,
        equity_curve,
        trade_pnls,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()

    config.apply_asset("NAS100")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = TIME_EXIT_BARS
    config.USE_MTF_CONFIRM = True

    df_all = _prepare_dataset("vantage_nas100.csv")
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


    # (El backtest real se ejecuta tras el entrenamiento óptimo, no aquí)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    best = study.best_trial.params

    # --- ENTRENAMIENTO FINAL CON LOS MEJORES PARÁMETROS (idéntico al test de $3,300) ---
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

    # Entrenamiento final en TODO el dataset de entrenamiento (como en el test validado)
    X_train_full, y_train_full = preparar_secuencias_flat_con_scaler(train_df, scaler)
    xgb_model, lgbm_model = _train_models(best_xgb, config.LGBM_PARAMS, X_train_full, y_train_full)

    # Backtest final para validar que la curva es idéntica
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

    print("\n✅ INFORME FINAL (NAS100 2025)")
    print(f"Beneficio Total: {final_result.profit:.2f}")
    print(f"Winrate Neto: {final_result.win_rate:.2f}%")
    print(f"Sharpe: {final_result.sharpe:.2f}")
    print(f"MaxDD %: {final_result.max_drawdown_pct:.2f}")
    print(f"Trades por día (promedio): {final_result.avg_trades_per_day:.2f}")

    # Exportar equity curve a CSV para análisis combinado
    import pandas as pd
    equity_df = pd.DataFrame(final_result.equity_curve, columns=["timestamp", "equity"])
    equity_df.to_csv("reports/equity_nas100_2024_2025.csv", index=False)

    # --- GUARDADO ROBUSTO DE MODELOS Y SCALER ---
    import joblib
    joblib.dump(xgb_model, "phoenix_nas100_xgb.pkl")
    joblib.dump(lgbm_model, "phoenix_nas100_lgbm.pkl")
    joblib.dump(scaler, "phoenix_nas100_scaler.pkl")
    print("\n✅ Modelos y scaler guardados: phoenix_nas100_xgb.pkl, phoenix_nas100_lgbm.pkl, phoenix_nas100_scaler.pkl")


if __name__ == "__main__":
    main()
