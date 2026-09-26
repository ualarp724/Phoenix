#!/usr/bin/env python3
"""
Sweep class-specific thresholds for BTCUSD weighted ensemble.
Goal: maximize avg daily with DD<15% and WinRate>60%.
"""

import pandas as pd
import numpy as np
import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from core.mtf import add_mtf_features_multi


def main():
    config.apply_asset("BTCUSD")
    config.ENSEMBLE_REQUIRE_CONSENSUS = False

    val_start = pd.to_datetime(config.VAL_START_DATE)
    val_end = pd.to_datetime(config.VAL_END_DATE)
    warmup_start = val_start - pd.Timedelta(days=60)

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare(
        date_start=warmup_start.strftime('%Y-%m-%d'),
        date_end=val_end.strftime('%Y-%m-%d'),
        apply_train_filter=False,
    )
    df = add_mtf_features_multi(df.copy(), config.MTF_CONFIGS)
    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[(df.index >= val_start) & (df.index <= val_end)].copy()

    backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)

    orig_buy = config.UMBRAL_BUY
    orig_sell = config.UMBRAL_SELL

    days = (val_end - val_start).days + 1
    thresholds = np.round(np.arange(0.30, 0.09, -0.01), 2)
    results = []

    for buy_th in thresholds:
        for sell_th in thresholds:
            config.UMBRAL_BUY = float(buy_th)
            config.UMBRAL_SELL = float(sell_th)
            result = backtester.backtest_period(df, precomputed_mtf=True)
            if not result:
                continue
            metrics = result['metrics']
            avg_daily = result['profit'] / days
            max_dd = metrics.get('max_drawdown_pct', 0.0)
            win_rate = metrics.get('win_rate', 0.0)

            if max_dd > 15.0 or win_rate < 60.0:
                continue

            results.append({
                'umbral_buy': float(buy_th),
                'umbral_sell': float(sell_th),
                'profit': result['profit'],
                'avg_daily': avg_daily,
                'win_rate': win_rate,
                'max_drawdown_pct': max_dd,
                'trades': metrics.get('total_trades', 0),
            })

    config.UMBRAL_BUY = orig_buy
    config.UMBRAL_SELL = orig_sell

    results = sorted(results, key=lambda x: x['avg_daily'], reverse=True)

    print("SWEEP BTC (DD<15%, WR>60%)")
    for r in results[:10]:
        print(r)

    print("\nBEST")
    print(results[0] if results else "No config met constraints")


if __name__ == "__main__":
    main()
