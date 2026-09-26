#!/usr/bin/env python3
"""Simulación forense multi-asset (Jul 2025 - Jan 2026)."""

from __future__ import annotations

import json
import warnings
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from core.mtf import add_mtf_features_multi

warnings.filterwarnings("ignore", category=UserWarning)

START_DATE = "2025-07-01"
END_DATE = "2026-01-31"
FIXED_LOT_SIZE = 0.01


@dataclass
class Position:
    side: int
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    spread_cost: float
    commission_cost: float


@dataclass
class AssetConfig:
    symbol: str
    data_path: str
    params: dict
    weight: float
    spread_pips: Optional[float] = None
    spread_usd: Optional[float] = None
    commission_per_lot: float = 0.01
    use_ema200: bool = True
    use_atr_filter: bool = True
    atr_ma_window: int = 100
    pip_value: Optional[float] = None  # USD per pip per 1.0 lot


@dataclass
class AssetState:
    cfg: AssetConfig
    model_xgb: Optional[object] = None
    model_lgbm: Optional[object] = None
    scaler: Optional[object] = None
    features: list[str] = None
    lookback: int = 60
    position: Optional[Position] = None


def _load_models_for_asset(symbol: str) -> tuple[Optional[object], Optional[object], Optional[object], list[str], int]:
    config.apply_asset(symbol)
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    model_xgb = None
    model_lgbm = None
    if Path(config.XGB_MODEL_PATH).exists():
        model_xgb = joblib.load(config.XGB_MODEL_PATH)
    if Path(config.LGBM_MODEL_PATH).exists():
        model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
    features = list(config.FEATURES)
    lookback = int(config.LOOKBACK_WINDOW)
    return model_xgb, model_lgbm, scaler, features, lookback


def _load_df(data_path: str) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        processor = PhoenixDataProcessor(data_path)
        df = processor.clean_and_prepare(apply_train_filter=False)
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)
    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna()
    if "EMA_200" not in df.columns:
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    return df


def _predict_signal(
    df: pd.DataFrame,
    state: AssetState,
    umbral_buy: float,
    umbral_sell: float,
) -> tuple[int, float, float]:
    for col in state.features:
        if col not in df.columns:
            df[col] = 0.0
    window = df[state.features].tail(state.lookback).values
    if len(window) < state.lookback:
        return 0, 0.0, float(df["Close"].iloc[-1])

    data_scaled = state.scaler.transform(window)
    flat = data_scaled.reshape(1, -1)

    if state.model_xgb is not None and state.model_lgbm is not None:
        probs_xgb = state.model_xgb.predict_proba(flat)
        probs_lgb = state.model_lgbm.predict_proba(flat)
        weights = config.ENSEMBLE_WEIGHTS
        combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)
    elif state.model_lgbm is not None:
        combined = state.model_lgbm.predict_proba(flat)
    elif state.model_xgb is not None:
        combined = state.model_xgb.predict_proba(flat)
    else:
        return 0, 0.0, float(df["Close"].iloc[-1])

    buy_score = float(combined[0][1])
    sell_score = float(combined[0][2])

    if buy_score >= sell_score:
        pred = 1 if buy_score >= umbral_buy else 0
        conf = buy_score if pred == 1 else 0.0
    else:
        pred = 2 if sell_score >= umbral_sell else 0
        conf = sell_score if pred == 2 else 0.0

    return pred, conf, float(df["Close"].iloc[-1])


def _calc_sl_tp(df: pd.DataFrame, price: float, sl_mult: float, tp_rr: float, side: int) -> tuple[float, float, float]:
    natr = float(df["NATR"].iloc[-1])
    atr = natr * price / 100.0
    sl_dist = max(atr * sl_mult, 0.0)
    tp_dist = sl_dist * tp_rr

    if side == 1:
        sl_price = price - sl_dist
        tp_price = price + tp_dist
    else:
        sl_price = price + sl_dist
        tp_price = price - tp_dist

    return sl_dist, sl_price, tp_price


def _apply_entry_costs(cfg: AssetConfig, price: float, lot_size: float, side: int) -> tuple[float, float, float]:
    spread_cost = 0.0
    if cfg.spread_pips is not None:
        spread_cost = cfg.spread_pips * (cfg.pip_value or 10.0) * (lot_size / config.MIN_LOT_SIZE)
    elif cfg.spread_usd is not None:
        spread_cost = cfg.spread_usd * (lot_size / config.MIN_LOT_SIZE)

    commission_cost = cfg.commission_per_lot * lot_size

    if cfg.spread_pips is not None:
        price = price + (cfg.spread_pips * 0.0001) if side == 1 else price - (cfg.spread_pips * 0.0001)
    elif cfg.spread_usd is not None:
        price = price + cfg.spread_usd if side == 1 else price - cfg.spread_usd

    return spread_cost, commission_cost, price


def _check_close(df: pd.DataFrame, pos: Position) -> tuple[bool, float]:
    hi = float(df["High"].iloc[-1])
    lo = float(df["Low"].iloc[-1])

    if pos.side == 1:
        if lo <= pos.sl_price and hi >= pos.tp_price:
            return True, pos.sl_price
        if lo <= pos.sl_price:
            return True, pos.sl_price
        if hi >= pos.tp_price:
            return True, pos.tp_price
    else:
        if hi >= pos.sl_price and lo <= pos.tp_price:
            return True, pos.sl_price
        if hi >= pos.sl_price:
            return True, pos.sl_price
        if lo <= pos.tp_price:
            return True, pos.tp_price

    return False, 0.0


def _should_trade(df: pd.DataFrame, cfg: AssetConfig, side: int) -> bool:
    if cfg.use_ema200:
        ema = float(df["EMA_200"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        if side == 1 and price <= ema:
            return False
        if side == 2 and price >= ema:
            return False

    if cfg.use_atr_filter:
        natr = float(df["NATR"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        atr = natr * price / 100.0
        atr_ma = float(df["NATR"].rolling(cfg.atr_ma_window).mean().iloc[-1]) * price / 100.0
        if atr <= atr_ma:
            return False

    return True


def _pnl_from_exit(cfg: AssetConfig, pos: Position, exit_price: float) -> float:
    if cfg.pip_value is not None:
        pips = (exit_price - pos.entry_price) / 0.0001 if pos.side == 1 else (pos.entry_price - exit_price) / 0.0001
        pnl = (pips * cfg.pip_value * pos.lot_size) - pos.spread_cost - pos.commission_cost
        return pnl
    pnl = (
        (exit_price - pos.entry_price) * pos.lot_size
        if pos.side == 1
        else (pos.entry_price - exit_price) * pos.lot_size
    )
    return pnl - pos.spread_cost - pos.commission_cost


def run_forensic(total_balance: float = 200.0) -> None:
    btc_params_path = Path("btc_final_params.json")
    eur_params_path = Path("eurusd_elite_v1.json")

    btc_params = json.loads(btc_params_path.read_text()) if btc_params_path.exists() else {}
    eur_params = json.loads(eur_params_path.read_text()) if eur_params_path.exists() else {}

    assets = [
        AssetConfig(
            symbol="XAUUSD",
            data_path="vantage_live_gold.csv",
            params={
                "umbral_buy": config.UMBRAL_BUY,
                "umbral_sell": config.UMBRAL_SELL,
                "sl_mult": config.ATR_SL_MULTIPLIER,
                "tp_rr": config.ATR_TP_MULTIPLIER,
            },
            weight=0.45,
            spread_usd=0.6,
            commission_per_lot=0.01,
            pip_value=None,
        ),
        AssetConfig(
            symbol="BTCUSD",
            data_path="vantage_btc2.csv",
            params={
                "umbral_buy": btc_params.get("umbral_buy", 0.20),
                "umbral_sell": btc_params.get("umbral_sell", 0.20),
                "sl_mult": btc_params.get("sl_mult", 2.0),
                "tp_rr": btc_params.get("tp_rr", 1.8),
            },
            weight=0.35,
            spread_usd=1.5,
            commission_per_lot=0.01,
            pip_value=None,
        ),
        AssetConfig(
            symbol="EURUSD",
            data_path="vantage_eurusd.csv",
            params={
                "umbral_buy": eur_params.get("umbral_buy", 0.148),
                "umbral_sell": eur_params.get("umbral_sell", 0.155),
                "sl_mult": eur_params.get("sl_mult", 2.28),
                "tp_rr": eur_params.get("tp_rr", 1.75),
            },
            weight=0.20,
            spread_pips=0.5,
            commission_per_lot=0.01,
            pip_value=10.0,
        ),
    ]

    states: dict[str, AssetState] = {}
    data_map: dict[str, pd.DataFrame] = {}

    for asset in assets:
        model_xgb, model_lgbm, scaler, features, lookback = _load_models_for_asset(asset.symbol)
        state = AssetState(
            cfg=asset,
            model_xgb=model_xgb,
            model_lgbm=model_lgbm,
            scaler=scaler,
            features=features,
            lookback=lookback,
        )
        df = _load_df(asset.data_path)
        df = df.loc[START_DATE:END_DATE].copy()
        data_map[asset.symbol] = df
        states[asset.symbol] = state

    all_times = sorted({t for df in data_map.values() for t in df.index})

    daily_pnl: dict[pd.Timestamp, dict[str, float]] = {}

    for ts in all_times:
        for symbol, state in states.items():
            df = data_map[symbol]
            if ts not in df.index:
                continue

            row = df.loc[:ts].iloc[-1:]
            if row.empty:
                continue

            # Cierre
            if state.position is not None:
                should_close, exit_price = _check_close(row, state.position)
                if should_close:
                    pnl = _pnl_from_exit(state.cfg, state.position, exit_price)
                    total_balance += pnl
                    outcome = "WIN" if pnl > 0 else "LOSS"
                    print(
                        f"[{ts}] - {symbol} - {'BUY' if state.position.side == 1 else 'SELL'} - "
                        f"{state.position.entry_price:.5f} - {outcome} - {pnl:+.2f}"
                    )
                    print(f"Saldo: ${total_balance:.2f}")

                    day = pd.Timestamp(ts.date())
                    daily_pnl.setdefault(day, {})
                    daily_pnl[day][symbol] = daily_pnl[day].get(symbol, 0.0) + pnl

                    state.position = None
                    continue

            if state.position is not None:
                continue

            pred, conf, price = _predict_signal(
                row,
                state,
                state.cfg.params["umbral_buy"],
                state.cfg.params["umbral_sell"],
            )

            if pred == 0:
                continue

            if not _should_trade(row, state.cfg, pred):
                continue

            sl_dist, sl_price, tp_price = _calc_sl_tp(row, price, state.cfg.params["sl_mult"], state.cfg.params["tp_rr"], pred)
            if sl_dist <= 0:
                continue

            lot_size = FIXED_LOT_SIZE
            if lot_size < config.MIN_LOT_SIZE:
                continue

            spread_cost, commission_cost, entry_price = _apply_entry_costs(state.cfg, price, lot_size, pred)

            state.position = Position(
                side=pred,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                spread_cost=spread_cost,
                commission_cost=commission_cost,
            )

    # Resumen de sinergia
    synergy_count = 0
    for day, pnl_map in daily_pnl.items():
        if len(pnl_map) < 2:
            continue
        positives = sum(p for p in pnl_map.values() if p > 0)
        negatives = sum(p for p in pnl_map.values() if p < 0)
        if positives > abs(negatives) and negatives < 0:
            synergy_count += 1

    print("\n📊 RESUMEN SINERGIA")
    print(f"Días con compensación positiva entre pares: {synergy_count}")


if __name__ == "__main__":
    run_forensic()
