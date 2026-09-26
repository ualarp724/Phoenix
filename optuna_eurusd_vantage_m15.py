def objective(trial, custom_params=None):
    import copy
    import phoenix_config as config
    from phoenix_processor import preparar_secuencias_flat_con_scaler
    from sklearn.preprocessing import StandardScaler
    # Configuración EURUSD
    config.apply_asset("EURUSD")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = 16
    config.USE_MTF_CONFIRM = True
    FEATURES_EURUSD = list(getattr(config, 'FEATURES', []))
    if not FEATURES_EURUSD:
        FEATURES_EURUSD = [
            'Open', 'High', 'Low', 'Close', 'Volume',
        ]
    df_all = _prepare_dataset("vantage_eurusd.csv")
    df_all = df_all.sort_index()
    train_df = df_all.loc["2022-01-01":"2023-12-31"].copy()
    val_df = df_all.loc["2024-01-01":"2025-12-31"].copy()
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES_EURUSD].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    params_lgbm = copy.deepcopy(getattr(config, 'LGBM_PARAMS', {}))
    if custom_params:
        sl_mult = custom_params.get("sl_mult", 2.0)
        tp_rr = custom_params.get("tp_rr", 1.5)
        umbral_buy = custom_params.get("umbral_buy", 0.15)
        umbral_sell = custom_params.get("umbral_sell", 0.15)
    else:
        sl_mult = trial.suggest_float("sl_mult", 2.0, 3.5)
        tp_rr = trial.suggest_float("tp_rr", 1.5, 2.2)
        umbral_buy = trial.suggest_float("umbral_buy", 0.10, 0.20)
        umbral_sell = trial.suggest_float("umbral_sell", 0.10, 0.20)
    lgbm_model = _train_models(params_lgbm, X_train, y_train)
    result = _backtest_multi_position(
        val_df,
        lgbm_model,
        scaler,
        umbral_buy,
        umbral_sell,
        sl_mult,
        tp_rr,
        weekday_only=False,
    )
    return result
from dataclasses import dataclass

@dataclass
class BacktestResult:
    profit: float
    win_rate: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    total_trades: int
    avg_trades_per_day: float
    equity_curve: list
    trade_pnls: list
    sharpe_ratio: float
# Añadir importación de pandas
import pandas as pd


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

    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna()
    return df


def _train_models(params_lgbm, X_train, y_train):
    tuned_lgbm = copy.deepcopy(params_lgbm)
    tuned_lgbm["n_jobs"] = 1
    lgbm_model = lgb.LGBMClassifier(**tuned_lgbm)
    lgbm_model.fit(X_train, y_train)
    return lgbm_model


def _mtf_trend_ok(row: pd.Series, pred: int) -> bool:
    m15_trend = float(row.get("M15_Trend_Score", 0.0))
    h1_trend = float(row.get("H1_Trend_Score", 0.0))
    if pred == 1:
        return m15_trend >= 0 and h1_trend >= 0
    if pred == 2:
        return m15_trend <= 0 and h1_trend <= 0
    return False


def _backtest_multi_position(
    val_df: pd.DataFrame,
    model_xgb,
    scaler,
    umbral_buy: float,
    umbral_sell: float,
    sl_mult: float,
    tp_rr: float,
    weekday_only: bool = False,
) -> BacktestResult:
    if val_df.empty:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, [], [], 0.0)

    data_scaled = scaler.transform(val_df[FEATURES_EURUSD].values)
    lookback = config.LOOKBACK_WINDOW
    if len(val_df) <= lookback + TIME_EXIT_BARS:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, [], [], 0.0)

    windows = np.lib.stride_tricks.sliding_window_view(data_scaled, lookback, axis=0)
    total_windows = windows.shape[0]
    flat = windows.reshape(total_windows, -1)

    probs_lgb = model_xgb.predict_proba(flat)
    combined = probs_lgb

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
    cooldown_until: pd.Timestamp | None = None
    returns: list[float] = []
    last_equity = capital

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
            sl_hit = False
            exit_price = None
            if side == 1:
                if lo <= sl_price and hi >= tp_price:
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif lo <= sl_price:
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif hi >= tp_price:
                    exit_price = tp_price
                    closed = True
            else:
                if hi >= sl_price and lo <= tp_price:
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif hi >= sl_price:
                    exit_price = sl_price
                    sl_hit = True
                    closed = True
                elif lo <= tp_price:
                    exit_price = tp_price
                    closed = True

            if not closed and (actual_idx - entry_idx) >= TIME_EXIT_BARS:
                exit_price = price
                closed = True

            if closed and exit_price is not None:
                if side == 1:
                    pips = (exit_price - entry_price) / PIP_VALUE
                else:
                    pips = (entry_price - exit_price) / PIP_VALUE
                pnl = (pips * 10 * lotes) - commission
                capital += pnl
                total_trades += 1
                if last_equity > 0:
                    returns.append(pnl / last_equity)
                last_equity = capital
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
                equity_curve.append((ts_now, capital))
                trade_pnls.append((ts_now, pnl))
                if sl_hit:
                    cooldown_until = ts_now + pd.Timedelta(hours=COOLDOWN_HOURS)
            else:
                still_open.append(pos)

        open_positions = still_open

        pred = int(preds[i])
        conf = float(confs[i])
        if pred == 0 or conf < MIN_CONF:
            i += 1
            continue

        if cooldown_until is not None and ts_now < cooldown_until:
            i += 1
            continue

        if weekday_only and val_df.index[actual_idx].weekday() >= 5:
            i += 1
            continue

        if actual_idx == last_entry_idx:
            i += 1
            continue


    # --- OBJETIVE EXTERNO PARA OPTIMIZACIÓN MULTI-ASSET ---

        i += 1

    # Cerrar posiciones restantes al final
    if open_positions:
        final_price = float(val_df["Close"].iloc[-1])
        final_ts = val_df.index[-1]
        for pos in open_positions:
            entry_price = pos["entry_price"]
            side = pos["side"]
            lotes = pos["lotes"]
            commission = pos["commission"]
            if side == 1:
                pips = (final_price - entry_price) / PIP_VALUE
            else:
                pips = (entry_price - final_price) / PIP_VALUE
            pnl = (pips * 10 * lotes) - commission
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

    if len(returns) >= 2:
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = (mean_ret / std_ret) * np.sqrt(len(returns)) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        profit,
        win_rate,
        max_dd,
        max_dd_pct,
        total_trades,
        avg_trades_per_day,
        equity_curve,
        trade_pnls,
        sharpe,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--audit-trial46", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine-high", action="store_true")
    args = parser.parse_args()

    config.apply_asset("EURUSD")
    config.TIMEFRAME = "M15"
    config.TARGET_LOOKAHEAD_BARS = TIME_EXIT_BARS
    config.USE_MTF_CONFIRM = True
    config.FEATURES = list(FEATURES_EURUSD)
    config.MIN_ATR_THRESHOLD = 0.0

    df_all = _prepare_dataset("vantage_eurusd.csv")
    df_all = df_all.sort_index()

    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("Split de datos inválido (train/validación vacíos).")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES_EURUSD].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    if len(X_train) != len(y_train):
        raise SystemExit("X_train/y_train desalineados.")


    if args.audit_trial46:
        best = TRIAL_46_PARAMS
    else:
        study = optuna.create_study(direction="maximize")
        if args.refine or args.refine_high:
            seed = dict(TRIAL_46_PARAMS)
            seed["sl_mult"] = max(0.8, min(seed["sl_mult"], 2.2))
            seed["tp_rr"] = max(0.5, min(seed["tp_rr"], 1.0))
            seed["umbral_buy"] = max(0.02, min(seed["umbral_buy"], 0.12))
            seed["umbral_sell"] = max(0.02, min(seed["umbral_sell"], 0.12))
            study.enqueue_trial(seed)
        try:
            study.optimize(objective, n_trials=args.trials, n_jobs=1)
        except KeyboardInterrupt:
            print("\n⚠️ Optimización interrumpida por el usuario.")
        best = study.best_trial.params
    best_lgbm = copy.deepcopy(config.LGBM_PARAMS)
    best_lgbm.update(
        {
            "max_depth": best["max_depth"],
            "n_estimators": best["n_estimators"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "num_leaves": best["num_leaves"],
            "min_data_in_leaf": best["min_data_in_leaf"],
            "min_gain_to_split": best["min_gain_to_split"],
            "max_bin": best["max_bin"],
            "verbose": -1,
        }
    )

    xgb_model = _train_models(best_lgbm, X_train, y_train)

    final_result = _backtest_multi_position(
        val_df,
        xgb_model,
        scaler,
        best["umbral_buy"],
        best["umbral_sell"],
        best["sl_mult"],
        best["tp_rr"],
        weekday_only=(args.refine or args.refine_high),
    )

    recovery = (
        final_result.profit / final_result.max_drawdown_usd
        if final_result.max_drawdown_usd > 0
        else float("inf")
    )

    print("\n✅ INFORME FINAL (EXAMEN 2024-2025)")
    print(f"Beneficio Total: {final_result.profit:.2f}")
    print(f"Winrate Neto: {final_result.win_rate:.2f}%")
    print(f"Factor de Recuperación Neto: {recovery:.2f}")
    print(f"Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
    print(f"Trades por día (promedio): {final_result.avg_trades_per_day:.2f}")

    with open(EURUSD_ELITE_V1_PATH, "w", encoding="utf-8") as f:
        json.dump(EURUSD_ELITE_V1, f, indent=2)

    report_result = final_result
    if args.refine:
        report_result = final_result

    if args.audit_trial46 or args.refine:
        if report_result.equity_curve:
            equity_df = pd.DataFrame(
                report_result.equity_curve, columns=["Timestamp", "Equity"]
            ).set_index("Timestamp")
            equity_df = equity_df.sort_index()
            equity_df = equity_df[~equity_df.index.duplicated(keep="last")]

            peak = equity_df["Equity"].cummax()
            drawdown_pct = (equity_df["Equity"] - peak) / peak * 100.0

            drawdowns = []
            in_dd = False
            dd_start = None
            dd_peak = None
            dd_min = None
            dd_min_time = None
            for ts, equity in equity_df["Equity"].items():
                equity_val = float(equity)
                current_peak = peak.loc[ts]
                if isinstance(current_peak, pd.Series):
                    current_peak = float(current_peak.iloc[-1])
                if equity_val < current_peak and not in_dd:
                    in_dd = True
                    dd_start = ts
                    dd_peak = current_peak
                    dd_min = equity_val
                    dd_min_time = ts
                elif in_dd:
                    if equity_val < dd_min:
                        dd_min = equity_val
                        dd_min_time = ts
                    if equity_val >= dd_peak:
                        duration = ts - dd_start
                        depth = (dd_peak - dd_min)
                        drawdowns.append(
                            (dd_start, dd_min_time, ts, depth, duration)
                        )
                        in_dd = False

            if in_dd and dd_start is not None and dd_peak is not None:
                duration = equity_df.index[-1] - dd_start
                depth = (dd_peak - dd_min) if dd_min is not None else 0.0
                drawdowns.append((dd_start, dd_min_time, equity_df.index[-1], depth, duration))

            drawdowns = sorted(drawdowns, key=lambda x: x[3], reverse=True)[:3]

            print("\n📉 TOP 3 DRAWDOWNS")
            for idx, (start, trough, recover, depth, duration) in enumerate(drawdowns, start=1):
                print(
                    f"{idx}. Inicio: {start.date()} | Mínimo: {trough.date()} | "
                    f"Recupera: {recover.date()} | Caída: ${depth:.2f} | "
                    f"Duración: {duration}"
                )

            pnl_series = pd.Series(
                {ts: pnl for ts, pnl in report_result.trade_pnls}
            )
            monthly_pnl = pnl_series.resample("ME").sum()

            print("\n📅 PNL NETO MENSUAL (2024-2025)")
            for ts, value in monthly_pnl.items():
                print(f"{ts.strftime('%Y-%m')}: {value:.2f}")

            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(equity_df.index, equity_df["Equity"], label="Equity")
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(drawdown_pct.index, drawdown_pct.values, color="red", alpha=0.5, label="Drawdown %")
                ax2.set_ylabel("Drawdown %")
                max_dd_pct = abs(drawdown_pct.min()) if not drawdown_pct.empty else 0.0
                plt.title(f"Equity Curve EURUSD (2024-2025) | Max DD: {max_dd_pct:.2f}%")
                plt.xlabel("Fecha")
                plt.ylabel("Capital (USD)")
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig("curva_equity_eurusd.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.plot(drawdown_pct.index, drawdown_pct.values, color="red")
                plt.title("Underwater (Drawdown %) EURUSD")
                plt.xlabel("Fecha")
                plt.ylabel("Drawdown %")
                plt.tight_layout()
                plt.savefig("underwater_eurusd.png")
                plt.close()
            except Exception as exc:  # pragma: no cover
                print(f"Error generando gráficos: {exc}")

    if args.refine:
        print("\n🧪 COMPARATIVA TRIAL 46 VS REFINADO")

        def _max_dd_pct(result: BacktestResult) -> float:
            if not result.equity_curve:
                return 0.0
            df = pd.DataFrame(result.equity_curve, columns=["Timestamp", "Equity"]).set_index("Timestamp")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            peak = df["Equity"].cummax()
            dd_pct = (df["Equity"] - peak) / peak * 100.0
            return abs(float(dd_pct.min())) if not dd_pct.empty else 0.0

        base_dd_pct = _max_dd_pct(report_result)
        refined_dd_pct = _max_dd_pct(final_result)

        print(
            f"Trial 46 -> Profit: {report_result.profit:.2f} | Winrate: {report_result.win_rate:.2f}% | "
            f"MaxDD%: {base_dd_pct:.2f}%"
        )
        print(
            f"Refinado -> Profit: {final_result.profit:.2f} | Winrate: {final_result.win_rate:.2f}% | "
            f"MaxDD%: {refined_dd_pct:.2f}%"
        )

        if refined_dd_pct >= base_dd_pct:
            print("Refinado NO supera al Trial 46 en estabilidad.")


if __name__ == "__main__":
    main()
