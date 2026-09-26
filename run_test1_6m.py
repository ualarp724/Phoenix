#!/usr/bin/env python3
"""
Run 6-month validation (Test 1) using ensemble consensus and report win rate >= 65%.
"""

import pandas as pd
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
    result = backtester.backtest_period(df, precomputed_mtf=True)

    if not result:
        print("No result for Test 1 period.")
        return

    metrics = result["metrics"]
    win_rate = metrics.get("win_rate", 0.0)

    print("TEST 1 (6 meses) RESULTADOS")
    print(f"Profit: ${result['profit']:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    days = (val_end - val_start).days + 1
    avg_daily = result['profit'] / days
    print(f"Avg Daily: ${avg_daily:.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
    print(f"Max Drawdown %: {metrics.get('max_drawdown_pct', 0.0):.2f}")
    print(f"✅ Win Rate >= 65%: {win_rate >= 65.0}")


if __name__ == "__main__":
    main()
