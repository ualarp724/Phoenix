"""Utilidades compartidas por los scripts de investigación (fase 3). Solo periodo de desarrollo."""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.backtest import run_backtest  # noqa: E402
from phoenix.data import dev_data, load_bars  # noqa: E402
from phoenix.features import FEATURES, build_features  # noqa: E402
from phoenix.metrics import monte_carlo_dd  # noqa: E402
from phoenix.settings import DATA_DIR, load_settings  # noqa: E402
from phoenix.walkforward import fit_side, make_windows, train_rows  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE / "experiments.csv"
COST_R = 0.10  # coste típico de una operación en R (0,28 $ sobre un SL de ~2-4 $)

S = load_settings()
bars = dev_data(load_bars(DATA_DIR / "vantage_gold.csv", S), S)
feats = build_features(bars)
windows = make_windows(S.splits.dev_start, S.splits.dev_end + pd.Timedelta(seconds=1))


def val_mask(w):
    return (bars.index >= w.val_start) & (bars.index < w.val_end)


def evaluate(orders: pd.DataFrame):
    """Backtest por ventana de validación; operaciones fuera de muestra concatenadas."""
    parts, skipped = [], 0
    for i, w in enumerate(windows):
        m = val_mask(w)
        res = run_backtest(bars[m], orders[m], S)
        skipped += res.skipped_min_lot
        if not res.trades.empty:
            parts.append(res.trades.assign(window=i))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), skipped


def stats(tr: pd.DataFrame, skipped: int) -> dict:
    if tr.empty:
        return {"trades": 0, "skipped_min_lot": skipped}
    r = tr["r"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [rng.choice(r, len(r)).mean() for _ in range(2000)]
    per_w = tr.groupby("window")["r"].mean()
    wins, losses = tr.pnl_usd[tr.pnl_usd > 0].sum(), -tr.pnl_usd[tr.pnl_usd <= 0].sum()
    months = sum((w.val_end - w.val_start).days for w in windows) / 30.44
    return {
        "trades": len(tr), "trades_per_month": len(tr) / months, "win_rate_pct": 100 * (r > 0).mean(),
        "avg_r": r.mean(), "avg_r_ci_low": float(np.quantile(boots, 0.025)), "avg_r_ci_high": float(np.quantile(boots, 0.975)),
        "profit_factor": wins / losses if losses else float("inf"),
        "windows_positive_pct": 100 * (per_w > 0).mean(), "pnl_usd": tr.pnl_usd.sum(),
        "mc_dd95_pct": monte_carlo_dd(tr.pnl_usd.to_numpy(), S.account.capital_usd), "skipped_min_lot": skipped,
        "per_window_avg_r": [round(x, 3) for x in per_w.reindex(range(len(windows))).fillna(0)],
    }


def log(name: str, cfg: dict, st: dict):
    """Apunta la configuración en experiments.csv (una sola vez por nombre + config)."""
    cfg_s = json.dumps(cfg, sort_keys=True)
    prev = pd.read_csv(EXPERIMENTS) if EXPERIMENTS.exists() else pd.DataFrame(columns=["name", "config"])
    if ((prev["name"] == name) & (prev["config"] == cfg_s)).any():
        return
    row = {"when": datetime.now().isoformat(timespec="seconds"), "name": name, "config": cfg_s,
           **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in st.items()}}
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(EXPERIMENTS, index=False)


def walkforward_proba(target: pd.Series) -> pd.Series:
    """Probabilidades fuera de muestra de LightGBM para `target`, ventana a ventana. Devuelve también el
    percentil de cada predicción respecto a las predicciones de SU conjunto de entrenamiento."""
    proba = pd.Series(np.nan, index=bars.index)
    pct = pd.Series(np.nan, index=bars.index)
    for w in windows:
        tr_m = train_rows(bars.index, w, purge_bars=32)
        model = fit_side(feats.loc[tr_m, FEATURES], target[tr_m])
        Xt = feats.loc[tr_m, FEATURES].dropna()
        train_pred = np.sort(model.predict_proba(Xt)[:, 1])
        Xv = feats.loc[val_mask(w), FEATURES].dropna()
        p = model.predict_proba(Xv)[:, 1]
        proba.loc[Xv.index] = p
        pct.loc[Xv.index] = np.searchsorted(train_pred, p) / len(train_pred)
    return proba, pct


def table(results: dict) -> list[str]:
    out = ["| Estrategia | Operaciones | Por mes | Acierto | R medio (IC 95 %) | PF | Ventanas en positivo | DD 95 % MC |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, st in results.items():
        if st.get("trades", 0) == 0:
            out.append(f"| {name} | 0 | | | | | | |")
            continue
        out.append(f"| {name} | {st['trades']} | {st['trades_per_month']:.0f} | {st['win_rate_pct']:.1f} % | "
                   f"{st['avg_r']:+.3f} ({st['avg_r_ci_low']:+.3f} a {st['avg_r_ci_high']:+.3f}) | {st['profit_factor']:.2f} | "
                   f"{st['windows_positive_pct']:.0f} % | {st['mc_dd95_pct']:.0f} % |")
    return out
