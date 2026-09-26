#!/usr/bin/env python3
"""Multi-asset master runner for XAUUSD, BTCUSD, EURUSD.

- Centralized risk: worst-case simultaneous losses <= $40.
- Higher weights for XAU/BTC, reduced for EUR.
- Prints total balance on each trade close.

This script is designed for live/paper integration using CSV feeds.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from core.mtf import add_mtf_features_multi


@dataclass
class Position:
    side: int  # 1=BUY, 2=SELL
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


@dataclass
class AssetState:
    cfg: AssetConfig
    model_xgb: Optional[object] = None
    model_lgbm: Optional[object] = None
    scaler: Optional[object] = None
    features: list[str] = field(default_factory=list)
    lookback: int = 60
    position: Optional[Position] = None


class CentralRiskManager:
    def __init__(self, total_balance: float, max_total_dd: float, weights: dict[str, float]):
        self.total_balance = total_balance
        self.max_total_dd = max_total_dd
        self.weights = weights

    def risk_budget_usd(self, symbol: str) -> float:
        weight = self.weights.get(symbol, 0.0)
        return self.max_total_dd * weight

    def lot_size_from_sl(self, symbol: str, sl_dist_usd: float) -> float:
        if sl_dist_usd <= 0:
            return 0.0
        risk_usd = self.risk_budget_usd(symbol)
        lot = risk_usd / sl_dist_usd
        lot = max(config.MIN_LOT_SIZE, min(lot, config.MAX_LOT_SIZE))
        return round(lot / 0.01) * 0.01


def _load_models_for_asset(symbol: str) -> tuple[Optional[object], Optional[object], Optional[object], list[str], int]:
    config.apply_asset(symbol)
    if not Path(config.SCALER_SAVE_PATH).exists():
        raise FileNotFoundError(config.SCALER_SAVE_PATH)
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


def _load_latest_df(data_path: str) -> pd.DataFrame:
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


def _apply_entry_costs(cfg: AssetConfig, price: float, lot_size: float, side: int) -> tuple[float, float]:
    spread_cost = 0.0
    if cfg.spread_pips is not None:
        spread_cost = cfg.spread_pips * 0.0001 * (lot_size / config.MIN_LOT_SIZE) * 10
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


def run_loop(total_balance: float = 200.0, max_total_dd: float = 40.0) -> None:
    btc_params_path = Path("btc_final_params.json")

    btc_params = json.loads(btc_params_path.read_text()) if btc_params_path.exists() else {}

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
        ),
        AssetConfig(
            symbol="NAS100",
            data_path="vantage_nas100.csv",
            params={
                "umbral_buy": config.UMBRAL_BUY,
                "umbral_sell": config.UMBRAL_SELL,
                "sl_mult": config.ATR_SL_MULTIPLIER,
                "tp_rr": config.ATR_TP_MULTIPLIER,
            },
            weight=0.20,
            spread_usd=1.0,
            commission_per_lot=0.01,
        ),
    ]

    weights = {a.symbol: a.weight for a in assets}
    risk_manager = CentralRiskManager(total_balance, max_total_dd, weights)

    states: dict[str, AssetState] = {}
    for asset in assets:
        try:
            model_xgb, model_lgbm, scaler, features, lookback = _load_models_for_asset(asset.symbol)
        except FileNotFoundError as exc:
            print(f"⚠️ {asset.symbol} omitido: falta {exc}")
            continue
        states[asset.symbol] = AssetState(
            cfg=asset,
            model_xgb=model_xgb,
            model_lgbm=model_lgbm,
            scaler=scaler,
            features=features,
            lookback=lookback,
        )

    print("✅ Multi-Asset Master iniciado")
    print(f"Balance total inicial: ${total_balance:.2f}")

    while True:
        for symbol, state in states.items():
            cfg = state.cfg
            if not Path(cfg.data_path).exists():
                continue

            config.apply_asset(symbol)
            df = _load_latest_df(cfg.data_path)
            if df.empty:
                continue

            # Cerrar posición si hay señales de SL/TP
            if state.position is not None:
                should_close, exit_price = _check_close(df, state.position)
                if should_close:
                    pos = state.position
                    pnl = (
                        (exit_price - pos.entry_price) * pos.lot_size
                        if pos.side == 1
                        else (pos.entry_price - exit_price) * pos.lot_size
                    )
                    pnl = pnl - pos.spread_cost - pos.commission_cost
                    total_balance += pnl
                    state.position = None
                    print(
                        f"[{symbol}] Cierre {'BUY' if pos.side == 1 else 'SELL'} | PnL: {pnl:+.2f} | "
                        f"Balance total: ${total_balance:.2f}"
                    )
                    continue

            if state.position is not None:
                continue

            pred, conf, price = _predict_signal(
                df,
                state,
                cfg.params["umbral_buy"],
                cfg.params["umbral_sell"],
            )

            if pred == 0:
                continue

            if not _should_trade(df, cfg, pred):
                continue

            sl_dist, sl_price, tp_price = _calc_sl_tp(df, price, cfg.params["sl_mult"], cfg.params["tp_rr"], pred)
            if sl_dist <= 0:
                continue

            lot_size = risk_manager.lot_size_from_sl(symbol, sl_dist)
            if lot_size < config.MIN_LOT_SIZE:
                continue

            spread_cost, commission_cost, entry_price = _apply_entry_costs(cfg, price, lot_size, pred)

            state.position = Position(
                side=pred,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                lot_size=lot_size,
                spread_cost=spread_cost,
                commission_cost=commission_cost,
            )

        time.sleep(30)


if __name__ == "__main__":
    run_loop()
