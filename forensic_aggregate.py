#!/usr/bin/env python3
"""Aggregated forensic simulation (Jul 2025 - Jan 2026) for XAUUSD, BTCUSD, EURUSD.

Runs each asset in isolation with its own parameters and filters, then
merges trades by close time to compute combined equity.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from core.mtf import add_mtf_features_multi

warnings.filterwarnings("ignore", category=UserWarning)

START_DATE = "2025-07-01"
END_DATE = "2026-01-31"
TIME_EXIT_BARS = 16


@dataclass
class Trade:
    close_time: pd.Timestamp
    symbol: str
    pnl: float
    entry_price: float
    sl_price: float
    tp_price: float
    lotes: float


@dataclass
class AssetParams:
    symbol: str
    data_path: str
    sl_mult: float
    tp_rr: float
    umbral_buy: float
    umbral_sell: float
    spread_pips: Optional[float] = None
    spread_usd: Optional[float] = None
    commission_per_lot: float = 0.01
    contract_size: Optional[float] = None
    use_ema200: bool = True
    use_atr_ma: bool = False
    atr_ma_window: int = 100
    min_atr_threshold: Optional[float] = None
    risk_usd: Optional[float] = None
    sl_usd: Optional[float] = None
    pip_value: Optional[float] = None


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


def _prepare_dataset_m15(csv_path: str) -> pd.DataFrame:
    df_m15 = _load_and_resample_m15(csv_path)
    df_m15.to_csv("__tmp_m15.csv", index=False)
    processor = PhoenixDataProcessor("__tmp_m15.csv")
    df = processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.0)
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


def _prepare_dataset_native(csv_path: str) -> pd.DataFrame:
    processor = PhoenixDataProcessor(csv_path)
    df = processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.0)
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


def _load_models(symbol: str) -> tuple[Optional[object], Optional[object], object]:
    config.apply_asset(symbol)
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model_xgb = joblib.load(config.XGB_MODEL_PATH) if Path(config.XGB_MODEL_PATH).exists() else None
    model_lgbm = joblib.load(config.LGBM_MODEL_PATH) if Path(config.LGBM_MODEL_PATH).exists() else None
    return model_xgb, model_lgbm, scaler


def _predict_probs(flat: np.ndarray, model_xgb, model_lgbm) -> np.ndarray:
    if model_xgb is not None and model_lgbm is not None:
        probs_xgb = model_xgb.predict_proba(flat)
        probs_lgb = model_lgbm.predict_proba(flat)
        weights = config.ENSEMBLE_WEIGHTS
        return (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)
    if model_lgbm is not None:
        return model_lgbm.predict_proba(flat)
    if model_xgb is not None:
        return model_xgb.predict_proba(flat)
    raise RuntimeError("No model available")


def _calc_pnl(
    params: AssetParams,
    side: int,
    entry_price: float,
    exit_price: float,
    lotes: float,
    spread_cost: float,
    commission: float,
) -> float:
    if params.pip_value is not None:
        pips = (exit_price - entry_price) / 0.0001 if side == 1 else (entry_price - exit_price) / 0.0001
        pnl = (pips * params.pip_value * lotes) - commission - spread_cost
        return pnl
    contract_size = params.contract_size or 1.0
    pnl = (exit_price - entry_price) * lotes * contract_size if side == 1 else (entry_price - exit_price) * lotes * contract_size
    return pnl - commission - spread_cost


def _backtest_asset(params: AssetParams, use_m15: bool) -> list[Trade]:
    config.apply_asset(params.symbol)
    model_xgb, model_lgbm, scaler = _load_models(params.symbol)

    df = _prepare_dataset_m15(params.data_path) if use_m15 else _prepare_dataset_native(params.data_path)
    df = df.sort_index().loc[START_DATE:END_DATE]
    if df.empty:
        return []

    data_scaled = scaler.transform(df[config.FEATURES].values)
    lookback = config.LOOKBACK_WINDOW
    if len(df) <= lookback + TIME_EXIT_BARS:
        return []

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    flat = windows.reshape(windows.shape[0], -1)
    probs = _predict_probs(flat, model_xgb, model_lgbm)

    buy_scores = probs[:, 1]
    sell_scores = probs[:, 2]
    preds = np.where(
        buy_scores >= sell_scores,
        np.where(buy_scores >= params.umbral_buy, 1, 0),
        np.where(sell_scores >= params.umbral_sell, 2, 0),
    )

    trades: list[Trade] = []
    open_pos = None

    for i in range(0, len(preds) - TIME_EXIT_BARS):
        idx = i + lookback
        row = df.iloc[idx]
        ts = df.index[idx]
        price = float(row["Close"])
        hi = float(row["High"])
        lo = float(row["Low"])

        # Close logic (pessimistic: SL first if both)
        if open_pos is not None:
            side = open_pos["side"]
            sl_price = open_pos["sl_price"]
            tp_price = open_pos["tp_price"]
            entry_price = open_pos["entry_price"]
            lotes = open_pos["lotes"]
            spread_cost = open_pos["spread_cost"]
            commission = open_pos["commission"]

            exit_price = None
            if side == 1:
                if lo <= sl_price and hi >= tp_price:
                    exit_price = sl_price
                elif lo <= sl_price:
                    exit_price = sl_price
                elif hi >= tp_price:
                    exit_price = tp_price
            else:
                if hi >= sl_price and lo <= tp_price:
                    exit_price = sl_price
                elif hi >= sl_price:
                    exit_price = sl_price
                elif lo <= tp_price:
                    exit_price = tp_price

            if exit_price is None and (idx - open_pos["entry_idx"]) >= TIME_EXIT_BARS:
                exit_price = price

            if exit_price is not None:
                pnl = _calc_pnl(params, side, entry_price, exit_price, lotes, spread_cost, commission)
                trades.append(
                    Trade(
                        close_time=ts,
                        symbol=params.symbol,
                        pnl=pnl,
                        entry_price=entry_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        lotes=lotes,
                    )
                )
                open_pos = None

        if open_pos is not None:
            continue

        pred = int(preds[i])
        if pred == 0:
            continue

        # Filters
        if params.use_ema200:
            ema = float(row["EMA_200"])
            if pred == 1 and price <= ema:
                continue
            if pred == 2 and price >= ema:
                continue

        atr = float(row["NATR"]) * price / 100.0
        if params.min_atr_threshold is not None and atr < params.min_atr_threshold:
            continue
        if params.use_atr_ma:
            atr_ma = float(df["NATR"].rolling(params.atr_ma_window).mean().iloc[idx]) * price / 100.0
            if atr <= atr_ma:
                continue

        sl_dist = atr * params.sl_mult
        if sl_dist <= 0:
            continue

        # lot sizing
        if params.sl_usd is not None:
            lotes = params.sl_usd / sl_dist
        elif params.risk_usd is not None and params.pip_value is not None:
            sl_pips = sl_dist / 0.0001
            lotes = params.risk_usd / (sl_pips * params.pip_value)
        elif params.risk_usd is not None and (params.contract_size or 0) > 0:
            lotes = params.risk_usd / (sl_dist * (params.contract_size or 1.0))
        else:
            lotes = config.MIN_LOT_SIZE

        if lotes < config.MIN_LOT_SIZE or lotes > config.MAX_LOT_SIZE:
            continue

        tp_dist = sl_dist * params.tp_rr
        if pred == 1:
            entry_price = price + (params.spread_usd or 0.0)
            if params.spread_pips is not None:
                entry_price = price + params.spread_pips * 0.0001
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            entry_price = price - (params.spread_usd or 0.0)
            if params.spread_pips is not None:
                entry_price = price - params.spread_pips * 0.0001
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        spread_cost = 0.0
        if params.spread_usd is not None:
            contract_size = params.contract_size or 1.0
            spread_cost = lotes * params.spread_usd * contract_size
        elif params.spread_pips is not None and params.pip_value is not None:
            spread_cost = params.spread_pips * params.pip_value * lotes

        commission = lotes * params.commission_per_lot

        open_pos = {
            "side": pred,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lotes": lotes,
            "entry_idx": idx,
            "spread_cost": spread_cost,
            "commission": commission,
        }

    return trades


def main() -> None:
    btc_params_path = Path("btc_final_params.json")
    if btc_params_path.exists():
        btc_params = json.loads(btc_params_path.read_text())
    else:
        fallback = Path("refined_params.json")
        if not fallback.exists():
            raise SystemExit("Falta btc_final_params.json y refined_params.json")
        btc_params = json.loads(fallback.read_text())
    eur_params = json.loads(Path("eurusd_elite_v1.json").read_text())

    xau_params = AssetParams(
        symbol="XAUUSD",
        data_path=config.ASSETS["XAUUSD"]["data_raw"],
        sl_mult=config.ATR_SL_MULTIPLIER,
        tp_rr=config.ATR_TP_MULTIPLIER,
        umbral_buy=config.UMBRAL_BUY,
        umbral_sell=config.UMBRAL_SELL,
        spread_usd=0.6,
        commission_per_lot=0.01,
        contract_size=100.0,
        use_ema200=True,
        use_atr_ma=False,
        min_atr_threshold=config.MIN_ATR_THRESHOLD,
        sl_usd=None,
        risk_usd=config.CAPITAL_INICIAL * config.RIESGO_POR_OPERACION,
        pip_value=None,
    )

    btc_params = AssetParams(
        symbol="BTCUSD",
        data_path="vantage_btc2.csv",
        sl_mult=btc_params.get("sl_mult", 2.3),
        tp_rr=btc_params.get("tp_rr", 1.8),
        umbral_buy=btc_params.get("umbral_buy", 0.20),
        umbral_sell=btc_params.get("umbral_sell", 0.20),
        spread_usd=1.5,
        commission_per_lot=6.0,
        use_ema200=True,
        use_atr_ma=False,
        min_atr_threshold=config.MIN_ATR_THRESHOLD,
        sl_usd=btc_params.get("sl_usd", 4.0),
        risk_usd=None,
        pip_value=None,
    )

    eur_params = AssetParams(
        symbol="EURUSD",
        data_path="vantage_eurusd.csv",
        sl_mult=eur_params["sl_mult"],
        tp_rr=eur_params["tp_rr"],
        umbral_buy=eur_params["umbral_buy"],
        umbral_sell=eur_params["umbral_sell"],
        spread_pips=0.5,
        commission_per_lot=0.01,
        use_ema200=True,
        use_atr_ma=True,
        atr_ma_window=100,
        min_atr_threshold=config.MIN_ATR_THRESHOLD,
        sl_usd=None,
        risk_usd=config.CAPITAL_INICIAL * 0.01,
        pip_value=10.0,
    )

    trades = []
    trades.extend(_backtest_asset(xau_params, use_m15=False))
    trades.extend(_backtest_asset(btc_params, use_m15=True))
    trades.extend(_backtest_asset(eur_params, use_m15=True))

    trades.sort(key=lambda t: t.close_time)

    xau_trades = [t for t in trades if t.symbol == "XAUUSD"]
    if xau_trades:
        print("\n🔎 ÚLTIMAS 5 OPERACIONES XAUUSD")
        for t in xau_trades[-5:]:
            print(
                f"[{t.close_time}] Entry={t.entry_price:.2f} SL={t.sl_price:.2f} "
                f"TP={t.tp_price:.2f} Lote={t.lotes:.2f} PnL={t.pnl:+.2f}"
            )

    if not trades:
        raise SystemExit("No trades generated. Filters too strict for the period.")

    balance = 200.0
    peak = balance
    max_dd = 0.0
    returns = []

    for t in trades:
        balance += t.pnl
        if balance > peak:
            peak = balance
        dd = peak - balance
        if dd > max_dd:
            max_dd = dd
        returns.append(t.pnl / peak if peak > 0 else 0.0)
        print(f"[{t.close_time}] {t.symbol} {t.pnl:+.2f} Saldo: ${balance:.2f}")

    sharpe = 0.0
    if len(returns) >= 2:
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / std_ret) * np.sqrt(len(returns)) if std_ret > 0 else 0.0

    print("\n✅ REPORTE FINAL")
    print(f"Beneficio Total: {balance - 200.0:.2f}")
    print(f"MaxDD: {max_dd:.2f}")
    print(f"Sharpe: {sharpe:.2f}")


if __name__ == "__main__":
    main()
