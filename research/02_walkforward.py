"""Fase 3, primera tanda: referencias (azar y reglas simples) contra LightGBM, en walk-forward.

Solo usa el periodo de desarrollo. Cada configuración se apunta en research/experiments.csv.
Uso (desde research/): python 02_walkforward.py  -> 02_walkforward.md
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd

from common import COST_R, HERE, S, bars, evaluate, feats, log, stats, table, windows
from phoenix.features import FEATURES
from phoenix.labels import tp_first_labels
from phoenix.strategies import (breakeven_prob, donchian_orders, london_breakout_orders, ml_orders,
                                random_orders)
from phoenix.walkforward import fit_side, train_rows

t0 = time.time()
results = {}


def run(name, cfg, orders, extra=None):
    tr, sk = evaluate(orders)
    results[name] = {**stats(tr, sk), **(extra or {})}
    log(name, cfg, results[name])


# 1) Referencias
for seed in range(5):
    run(f"Azar (semilla {seed})", {"strategy": "random", "seed": seed, "sl_k": 1, "tp_m": 1.5, "max_bars": 16},
        random_orders(bars, feats, seed=seed))
run("Ruptura Donchian 32 + EMA200", {"strategy": "donchian", "sl_k": 1, "tp_m": 1.5, "max_bars": 16}, donchian_orders(bars, feats))
run("Ruptura rango de Londres", {"strategy": "london", "sl_k": 1, "tp_m": 1.5, "max_bars": 16}, london_breakout_orders(bars, feats))

# 2) LightGBM: un clasificador por lado que predice «llega al TP antes que al SL»
top_features = None
for sl_k, tp_m, max_bars, margin in [(1.0, 1.5, 16, 0.05), (1.0, 2.0, 32, 0.05), (1.0, 1.5, 16, 0.10)]:
    labels = tp_first_labels(bars, sl_k * feats["atr"], tp_m * sl_k * feats["atr"], max_bars, S)
    p = {"long": pd.Series(np.nan, index=bars.index), "short": pd.Series(np.nan, index=bars.index)}
    imps = []
    for w in windows:
        tr_m = train_rows(bars.index, w, purge_bars=max_bars)
        Xv = feats.loc[(bars.index >= w.val_start) & (bars.index < w.val_end), FEATURES].dropna()
        for side in ("long", "short"):
            model = fit_side(feats.loc[tr_m, FEATURES], labels.loc[tr_m, side])
            p[side].loc[Xv.index] = model.predict_proba(Xv)[:, 1]
            imps.append(pd.Series(model.booster_.feature_importance("gain"), index=FEATURES))
    thr = breakeven_prob(tp_m, COST_R) + margin
    name = f"LightGBM SL {sl_k:g} ATR, TP {tp_m:g}x, {max_bars} velas, umbral {thr:.2f}"
    run(name, {"strategy": "lgbm_tp_first", "sl_k": sl_k, "tp_m": tp_m, "max_bars": max_bars, "threshold": round(thr, 4)},
        ml_orders(bars, feats, p["long"], p["short"], threshold=thr, sl_k=sl_k, tp_m=tp_m, max_bars=max_bars))
    if top_features is None:
        imp = pd.concat(imps, axis=1).mean(axis=1)
        top_features = (imp / imp.sum()).sort_values(ascending=False).head(8)

out = ["# Fase 3 — Primera tanda: walk-forward en desarrollo (XAUUSD M15)\n",
       f"Generado por `research/02_walkforward.py` el {datetime.now():%d/%m/%Y}. {len(windows)} ventanas de validación de 3 meses "
       f"({windows[0].val_start:%d/%m/%Y} → {windows[-1].val_end:%d/%m/%Y}), cada una entrenada con los 18 meses anteriores. "
       f"SL = 1 ATR (máx. {S.account.max_risk_usd:.2f} $/oz por el lote mínimo), riesgo fijo de {S.account.max_risk_usd:.2f} $, costes reales.\n"]
out += table(results)
out.append("\n## R medio por ventana\n")
out.append("| Estrategia | " + " | ".join(f"{w.val_start:%m/%y}" for w in windows) + " |")
out.append("| --- |" + " --- |" * len(windows))
for name, st in results.items():
    if st.get("trades", 0):
        out.append(f"| {name} | " + " | ".join(f"{x:+.2f}" for x in st["per_window_avg_r"]) + " |")
out.append("\n## Features más usadas por LightGBM (SL 1 ATR, TP 1,5x)\n")
out.append("| Feature | Peso |\n| --- | --- |")
out += [f"| `{k}` | {100 * v:.1f} % |" for k, v in top_features.items()]
out.append(f"\nTiempo de cálculo: {time.time() - t0:.0f} s.")
(HERE / "02_walkforward.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
