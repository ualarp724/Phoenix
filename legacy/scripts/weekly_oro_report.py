import pandas as pd
import numpy as np
import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from core.mtf import add_mtf_features_multi
from backtest_oro_stress_test import OroStressBacktester, load_best_params, compute_fixed_threshold


def main():
    config.apply_asset("XAUUSD")
    params = load_best_params()
    processor = PhoenixDataProcessor("m5.csv")
    df_base = processor.clean_and_prepare(apply_train_filter=False)
    df_full = add_mtf_features_multi(df_base.copy(), config.MTF_CONFIGS).dropna()
    backtester = OroStressBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    fixed = compute_fixed_threshold(backtester, df_full)

    start = pd.Timestamp("2025-08-01")
    end = df_full.index.max()
    df_period = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()
    result = backtester.backtest_period(
        df_period,
        params,
        precomputed_mtf=True,
        fixed_threshold=fixed,
        reset_capital=True,
    )

    trades = pd.DataFrame(result["trades_log"])
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades = trades.dropna(subset=["timestamp"]).sort_values("timestamp")
    capital_history = result["capital_history"]
    if len(capital_history) == len(trades) + 1:
        trades["capital_before"] = capital_history[:-1]
        trades["capital_after"] = capital_history[1:]
    else:
        trades["capital_before"] = (
            config.CAPITAL_INICIAL + trades["pnl"].cumsum().shift(fill_value=0)
        )
        trades["capital_after"] = config.CAPITAL_INICIAL + trades["pnl"].cumsum()

    trades["week"] = trades["timestamp"].dt.to_period("W-MON")
    rows = []
    init_cap = config.CAPITAL_INICIAL

    for period, tdf in trades.groupby("week"):
        week_start = period.start_time
        week_end = period.end_time
        prev = trades[trades["timestamp"] < week_start]
        cap_start = float(prev["capital_after"].iloc[-1]) if len(prev) else init_cap
        pnl = float(tdf["pnl"].sum())
        cap_end = cap_start + pnl
        wins = (tdf["pnl"] > 0).sum()
        losses = (tdf["pnl"] < 0).sum()
        total = len(tdf)
        winrate = float(wins / total * 100) if total else 0.0
        gross_win = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        profit_factor = (
            float(gross_win / abs(gross_loss))
            if gross_loss < 0
            else (float("inf") if gross_win > 0 else 0.0)
        )

        equity = [cap_start]
        equity.extend((cap_start + tdf["pnl"].cumsum()).tolist())
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        rows.append(
            {
                "week_start": week_start.date(),
                "week_end": week_end.date(),
                "trades": total,
                "wins": int(wins),
                "losses": int(losses),
                "winrate_pct": round(winrate, 2),
                "pnl_usd": round(pnl, 2),
                "capital_start": round(cap_start, 2),
                "capital_end": round(cap_end, 2),
                "max_dd_pct": round(max_dd * 100, 2),
                "profit_factor": round(profit_factor, 2),
            }
        )

    weekly = pd.DataFrame(rows).sort_values("week_start")
    out_path = "oro_weekly_2025-08_to_latest.csv"
    weekly.to_csv(out_path, index=False)

    print(f"Rango: {start.date()} -> {end.date()}")
    print(f"Semanas: {len(weekly)}")
    print(weekly.tail(5).to_string(index=False))
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()
