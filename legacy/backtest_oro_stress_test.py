import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_metrics import TradingMetrics
from core.execution_simulator import apply_execution_costs
from core.logging_utils import log_event
from core.regime import entry_allowed
from core.mtf import add_mtf_features_multi, mtf_confirm
from core.risk_manager import RiskManager
from core.execution_engine_stub import ExecutionEngine, OrderRequest
from core.news_checker import NewsFilter
from core.data_validators import validate_features, validate_no_nulls, validate_target


SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.0
COMMISSION_PER_LOT = 7.0  # $7 por lote estándar
MIN_ROWS_PER_MONTH = 800
OUTPUT_IMAGE = "equity_oro_stress_test.png"
ENABLE_CIRCUIT_BREAKER = False


@dataclass
class BestParams:
    umbral: float
    sl_mult: float
    tp_mult: float
    use_time_filter: bool
    hour_start: int
    hour_end: int


def load_best_params(path: str = "optuna_best.json") -> BestParams:
    default = BestParams(
        umbral=config.UMBRAL_CONFIANZA,
        sl_mult=config.ATR_SL_MULTIPLIER,
        tp_mult=config.ATR_TP_MULTIPLIER,
        use_time_filter=config.USE_TIME_FILTER,
        hour_start=config.HORA_INICIO,
        hour_end=config.HORA_CIERRE,
    )
    if not Path(path).exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("params", {})
        return BestParams(
            umbral=float(params.get("threshold", default.umbral)),
            sl_mult=float(params.get("sl_mult", default.sl_mult)),
            tp_mult=float(params.get("tp_mult", default.tp_mult)),
            use_time_filter=bool(params.get("use_time_filter", default.use_time_filter)),
            hour_start=int(params.get("hour_start", default.hour_start)),
            hour_end=int(params.get("hour_end", default.hour_end)),
        )
    except Exception:
        return default


def resolve_feature_columns(df: pd.DataFrame, scaler) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        cols = [c for c in scaler.feature_names_in_ if c in df.columns]
    else:
        cols = [c for c in config.FEATURES if c in df.columns]
    if not cols:
        raise ValueError("No se encontraron columnas de features compatibles.")
    return cols


def compute_fixed_threshold(backtester: "OroStressBacktester", df_full: pd.DataFrame) -> float:
    df_full = df_full.dropna()
    feature_cols = resolve_feature_columns(df_full, backtester.scaler)
    data_scaled = backtester.scaler.transform(df_full[feature_cols].values)
    lookback = config.LOOKBACK_WINDOW
    if len(df_full) <= lookback + 20:
        raise RuntimeError("No hay suficientes datos para calcular umbral fijo.")
    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)
    batch_size = getattr(config, "PRED_BATCH_SIZE", 512)
    confidences = np.empty(total_windows, dtype=np.float32)
    for start in range(0, total_windows, batch_size):
        end = min(start + batch_size, total_windows)
        if backtester.model_type == "ensemble" and backtester.model_lgbm is not None:
            probs_xgb = backtester.model.predict_proba(flat[start:end])
            probs_lgb = backtester.model_lgbm.predict_proba(flat[start:end])
            weights = config.ENSEMBLE_WEIGHTS
            if config.ENSEMBLE_REQUIRE_CONSENSUS:
                preds_xgb = np.argmax(probs_xgb, axis=1)
                preds_lgb = np.argmax(probs_lgb, axis=1)
                confs_xgb = np.max(probs_xgb, axis=1)
                confs_lgb = np.max(probs_lgb, axis=1)
                consensus = (preds_xgb == preds_lgb) & (preds_xgb != 0)
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
            probs = backtester.model.predict_proba(flat[start:end])
            confs = np.max(probs, axis=1)
        confidences[start:end] = confs
    conf_np = confidences[confidences > 0]
    if conf_np.size == 0:
        conf_np = confidences
    return float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))


class OroStressBacktester:
    def __init__(self, model_path: str, scaler_path: str):
        self.model_type = config.MODEL_TYPE
        self.model = joblib.load(model_path)
        self.model_lgbm = None
        if self.model_type == "ensemble":
            self.model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        self.scaler = joblib.load(scaler_path)
        self.news_filter = NewsFilter()

    def backtest_period(
        self,
        df_period: pd.DataFrame,
        params: BestParams,
        precomputed_mtf: bool = True,
        fixed_threshold: float | None = None,
        reset_capital: bool = True,
    ):
        init_capital = config.CAPITAL_INICIAL if reset_capital else config.CAPITAL_INICIAL
        capital = init_capital
        capital_history = [capital]
        trades_log = []
        peak_capital = capital

        if not precomputed_mtf:
            df_period = add_mtf_features_multi(df_period.copy(), config.MTF_CONFIGS)
        df_period = df_period.dropna()
        validate_features(df_period)
        validate_no_nulls(df_period)
        validate_target(df_period)

        feature_cols = resolve_feature_columns(df_period, self.scaler)
        data_scaled = self.scaler.transform(df_period[feature_cols].values)
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

        if fixed_threshold is None:
            conf_np = confidences[confidences > 0]
            if conf_np.size == 0:
                conf_np = confidences
            dyn_threshold = float(np.quantile(conf_np, config.CONFIDENCE_PERCENTILE))
        else:
            dyn_threshold = float(fixed_threshold)

        risk_manager = RiskManager(account_balance=capital)
        exec_engine = ExecutionEngine(risk_manager)

        for i in range(total_windows - 20):
            if capital < config.CAPITAL_PROTECCIÓN:
                break

            actual_idx = i + lookback
            if ENABLE_CIRCUIT_BREAKER:
                if capital > peak_capital:
                    peak_capital = capital
                max_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
                if max_dd >= 0.20:
                    break

            try:
                current_ts = df_period.index[actual_idx]
                h = current_ts.hour if hasattr(current_ts, "hour") else 12
            except Exception:
                current_ts = None
                h = 12

            if params.use_time_filter and (h < params.hour_start or h >= params.hour_end):
                continue
            if current_ts is not None and self.news_filter.high_impact_at(current_ts):
                continue

            conf = float(confidences[i])
            pred = int(predictions[i])

            if pred != 0 and conf > max(params.umbral, dyn_threshold):
                row = df_period.iloc[actual_idx]
                price = row["Close"]
                atr = df_period["NATR"].iloc[actual_idx] * price / 100

                if actual_idx > 0:
                    prev_close = df_period["Close"].iloc[actual_idx - 1]
                    body = abs(price - prev_close)
                    wick = max(row["High"] - row["Low"] - body, 0.0)
                    if body == 0 or wick > (0.5 * body):
                        continue

                atr_series = (df_period["NATR"] * df_period["Close"] / 100.0)
                atr_last3 = atr_series.iloc[max(actual_idx - 2, 0): actual_idx + 1].mean()
                atr_daily_avg = atr_series.iloc[max(actual_idx - 288, 0): actual_idx + 1].mean()
                if risk_manager.block_flash_crash(atr_last3, atr_daily_avg):
                    continue

                if row["Vol_Rel"] < config.MIN_VOL_REL:
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

                sl_dist = atr * params.sl_mult
                tp_dist = atr * params.tp_mult

                target_mult = (
                    (config.TARGET_DAILY_USD / config.EXPECTED_DAILY_USD)
                    if config.EXPECTED_DAILY_USD
                    else 1.0
                )
                riesgo_pct = config.RIESGO_POR_OPERACION * target_mult * config.RISK_MULTIPLIER
                riesgo_pct = min(config.MAX_RISK_PER_TRADE_PCT, riesgo_pct)
                riesgo = capital * riesgo_pct
                lotes = max(riesgo / (sl_dist * 100), 0.01)
                lotes = max(config.MIN_LOT_SIZE, min(lotes, config.MAX_LOT_SIZE))

                risk_manager.account_balance = capital
                order_id = exec_engine.submit_order(
                    OrderRequest(
                        symbol=config.SYMBOL,
                        side="BUY" if pred == 1 else "SELL",
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

                if pred == 1:
                    tp_price = price + tp_dist
                    sl_price = price - sl_dist
                else:
                    tp_price = price - tp_dist
                    sl_price = price + sl_dist

                pnl = 0.0
                for j in range(1, 17):
                    if actual_idx + j >= len(df_period):
                        break
                    hi = df_period["High"].iloc[actual_idx + j]
                    lo = df_period["Low"].iloc[actual_idx + j]

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

                if pnl == 0 and actual_idx + 16 < len(df_period):
                    exit_p = df_period["Close"].iloc[actual_idx + 16]
                    pnl = (exit_p - price) * 100 * lotes if pred == 1 else (price - exit_p) * 100 * lotes

                exec_side = "BUY" if pred == 1 else "SELL"
                exec_result = apply_execution_costs(
                    price,
                    lotes,
                    exec_side,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                )
                comision = lotes * COMMISSION_PER_LOT
                neto = pnl - comision - exec_result.spread_usd - exec_result.slippage_usd

                capital += neto
                risk_manager.record_trade_closed(neto)
                capital_history.append(capital)

                trades_log.append(
                    {
                        "entry_price": price,
                        "exit_price": price,
                        "direction": "BUY" if pred == 1 else "SELL",
                        "size": lotes,
                        "pnl": neto,
                        "timestamp": current_ts,
                    }
                )
                log_event(
                    {
                        "event": "BACKTEST_TRADE",
                        "direction": "BUY" if pred == 1 else "SELL",
                        "pnl": neto,
                        "sl_dist": sl_dist,
                        "tp_dist": tp_dist,
                    }
                )

        if len(trades_log) == 0:
            return None

        metrics = TradingMetrics(init_capital, trades_log)
        summary = {
            "capital_final": capital,
            "profit": capital - init_capital,
            "trades": len(trades_log),
            "capital_history": capital_history,
            "trades_log": trades_log,
            "metrics": metrics.generar_reporte_completo(np.array(capital_history)),
        }
        return summary


def month_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["month"] = df.index.to_period("M")
    for period, mdf in df.groupby("month"):
        if len(mdf) < MIN_ROWS_PER_MONTH:
            continue
        start = float(mdf["Close"].iloc[0])
        end = float(mdf["Close"].iloc[-1])
        trend = abs(end - start) / start if start else 0.0
        rows.append(
            {
                "month": period,
                "mean_natr": float(mdf["NATR"].mean()),
                "trend_strength": float(trend),
                "rows": int(len(mdf)),
            }
        )
    return pd.DataFrame(rows)


def pick_months(stats: pd.DataFrame) -> dict:
    if stats.empty or len(stats) < 3:
        raise RuntimeError("No hay meses suficientes para el stress test.")

    vol_q = stats["mean_natr"].quantile(0.75)
    trend_q = stats["trend_strength"].quantile(0.75)
    low_trend_q = stats["trend_strength"].quantile(0.25)
    mid_vol_q = stats["mean_natr"].quantile(0.5)

    high_vol = stats[stats["mean_natr"] >= vol_q]
    lateral = stats[(stats["trend_strength"] <= low_trend_q) & (stats["mean_natr"] <= mid_vol_q)]
    trend = stats[stats["trend_strength"] >= trend_q]

    chosen = {}
    used = set()

    def choose(label, df):
        candidates = df[~df["month"].isin(used)]
        if candidates.empty:
            candidates = stats[~stats["month"].isin(used)]
        pick = candidates.sample(n=1, random_state=random.randint(1, 10_000)).iloc[0]
        used.add(pick["month"])
        chosen[label] = pick

    choose("alta_volatilidad", high_vol)
    choose("lateral", lateral)
    choose("tendencia", trend)
    return chosen


def month_range(period: pd.Period):
    start = period.to_timestamp(how="start")
    end = (period + 1).to_timestamp(how="start") - pd.Timedelta(minutes=1)
    return start, end


def build_equity_series(month_start, trades_log, capital_history):
    timestamps = [month_start]
    timestamps.extend([t["timestamp"] for t in trades_log])
    if len(timestamps) != len(capital_history):
        timestamps = list(range(len(capital_history)))
    return pd.Series(capital_history, index=timestamps)


def slice_result_to_period(
    full_result: dict,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    init_capital: float,
):
    trades = pd.DataFrame(full_result.get("trades_log", []))
    if trades.empty:
        return None
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades = trades.dropna(subset=["timestamp"]).sort_values("timestamp")

    capital_history = full_result.get("capital_history", [])
    if len(capital_history) == len(trades) + 1:
        trades["capital_before"] = capital_history[:-1]
        trades["capital_after"] = capital_history[1:]
    else:
        trades["capital_before"] = init_capital + trades["pnl"].cumsum().shift(fill_value=0)
        trades["capital_after"] = init_capital + trades["pnl"].cumsum()

    trades_before = trades[trades["timestamp"] < period_start]
    if trades_before.empty:
        capital_at_start = init_capital
    else:
        capital_at_start = float(trades_before["capital_after"].iloc[-1])

    trades_in = trades[
        (trades["timestamp"] >= period_start) & (trades["timestamp"] <= period_end)
    ].copy()
    if trades_in.empty:
        return None

    pnl_sum = float(trades_in["pnl"].sum())
    capital_end = capital_at_start + pnl_sum

    capital_history_month = [capital_at_start]
    capital_history_month.extend((capital_at_start + trades_in["pnl"].cumsum()).tolist())

    trades_log_month = trades_in.drop(
        columns=["capital_before", "capital_after"], errors="ignore"
    ).to_dict("records")

    metrics = TradingMetrics(capital_at_start, trades_log_month)
    summary = {
        "capital_final": capital_end,
        "profit": pnl_sum,
        "trades": len(trades_log_month),
        "capital_history": capital_history_month,
        "trades_log": trades_log_month,
        "metrics": metrics.generar_reporte_completo(np.array(capital_history_month)),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="Mes específico YYYY-MM")
    parser.add_argument("--fixed-threshold", action="store_true", help="Usar umbral fijo global")
    parser.add_argument(
        "--context-start",
        help="Inicio de contexto para warmup (YYYY-MM-DD), útil para igualar meses dentro de rangos",
    )
    parser.add_argument(
        "--context-end",
        help="Fin de contexto para warmup (YYYY-MM-DD), por defecto fin del mes",
    )
    parser.add_argument(
        "--total-profit",
        action="store_true",
        help="Calcular profit total en rango 2024-01-01 a 2026-02-06",
    )
    args = parser.parse_args()

    config.apply_asset("XAUUSD")
    params = load_best_params()

    data_file = Path("m5.csv")
    if not data_file.exists():
        raise FileNotFoundError("No se encontró m5.csv para Oro (no se usa M15).")

    processor = PhoenixDataProcessor(str(data_file))
    df_base = processor.clean_and_prepare(apply_train_filter=False)
    df_full = add_mtf_features_multi(df_base.copy(), config.MTF_CONFIGS)
    df_full = df_full.dropna()

    backtester = OroStressBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    fixed_threshold = None
    if args.fixed_threshold:
        fixed_threshold = compute_fixed_threshold(backtester, df_full)

    if args.total_profit:
        # Filtrar rango 2024-01-01 a 2026-02-06
        start = pd.to_datetime("2024-01-01")
        end = pd.to_datetime("2026-02-06")
        df_range = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()
        result = backtester.backtest_period(
            df_range,
            params,
            precomputed_mtf=True,
            fixed_threshold=fixed_threshold,
            reset_capital=True,
        )
        if not result:
            print("Sin trades en el rango 2024-2026.")
            return
        metrics = result["metrics"]
        print("\n=== PROFIT TOTAL XAUUSD M5 (2024-2026) ===")
        print(f"Profit: ${result['profit']:.2f}")
        print(f"Winrate: {metrics['win_rate']:.2f}%")
        print(f"Max DD: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Trades: {result['trades']}")
        return

    # ...existing code...


if __name__ == "__main__":
    main()
