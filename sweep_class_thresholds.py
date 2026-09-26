#!/usr/bin/env python3
"""
Sweep class-specific thresholds for weighted ensemble.
Targets: maximize avg daily, DD<=15%, WinRate>=60%, and trades >= 2x baseline.
"""

import pandas as pd
import numpy as np
import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_backtester_pro import ProfessionalBacktester
from core.mtf import add_mtf_features_multi


def main():
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
    df = df[(df.index >= val_start) & (df.index <= val_end)].copy()

    backtester = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)

    # Baseline trades with current thresholds
    baseline = backtester.backtest_period(df, precomputed_mtf=True)
    if not baseline:
        print("No baseline result.")
        return
    baseline_trades = baseline['metrics'].get('total_trades', 0)
    target_trades = baseline_trades * 2

    days = (val_end - val_start).days + 1

    thresholds = np.round(np.arange(0.24, 0.05, -0.01), 2)
    results = []

    orig_buy = config.UMBRAL_BUY
    orig_sell = config.UMBRAL_SELL

    for buy_th in thresholds:
        for sell_th in thresholds:
            config.UMBRAL_BUY = float(buy_th)
            config.UMBRAL_SELL = float(sell_th)
            result = backtester.backtest_period(
                df,
                precomputed_mtf=True,
                umbral=config.UMBRAL_CONFIANZA,
            )
            if not result:
                continue

            metrics = result['metrics']
            trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0.0)
            max_dd = metrics.get('max_drawdown_pct', 0.0)
            avg_daily = result['profit'] / days

            if win_rate < 60.0:
                continue
            if max_dd > 15.0:
                continue

            results.append({
                'umbral_buy': float(buy_th),
                'umbral_sell': float(sell_th),
                'profit': result['profit'],
                'avg_daily': avg_daily,
                'win_rate': win_rate,
                'max_drawdown_pct': max_dd,
                'trades': trades,
            })

    config.UMBRAL_BUY = orig_buy
    config.UMBRAL_SELL = orig_sell

    results = sorted(results, key=lambda x: x['avg_daily'], reverse=True)

    print(f"BASELINE TRADES: {baseline_trades}")
    print(f"TARGET TRADES: {target_trades}")
    print("SWEEP RESULTS (DD<=15%, WR>=60%)")
    for r in results[:10]:
        print(r)

    print("\nBEST")
    print(results[0] if results else "No config met constraints")
    if results:
        best = results[0]
        print(f"\nDOUBLED TRADES? {best['trades'] >= target_trades}")


if __name__ == "__main__":
    main()
