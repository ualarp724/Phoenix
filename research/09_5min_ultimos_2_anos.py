"""Idea de Arturo: operar en velas de 5 minutos usando solo los 2 últimos años (el mercado ha cambiado).

Declarado antes de ejecutar. Mismo simulador (9x, comisión taker 0,05 %, funding, liquidación),
mismo objetivo (100 € → 500 € en 30 días). Ventanas que empiezan cada día del 01/10/2024 al 26/08/2026.
  F1 5 min: ruptura 20/10, stop 2 ATR, riesgo 30 %, filtro de tendencia (EMA 300 velas ≈ 25 h), piramidar
  F2 5 min: ruptura 48/20, resto igual que F1
  F3 5 min: F1 sin piramidar
  F4 5 min: control al azar con el tamaño de F1 (3 semillas)
  S4 4 h: la estrategia del bot actual, en las mismas ventanas, para comparar
Uso (desde la raíz): python research/09_5min_ultimos_2_anos.py
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.btc5m import binance  # noqa: E402
from phoenix.perps import data  # noqa: E402
from phoenix.perps.sim import Costs, run  # noqa: E402
from phoenix.perps.strategies import add_indicators, breakout, load_4h, random_entries, resample  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPITAL, TARGET, LEV = 100.0, 500.0, 9.0
COSTS = Costs(max_leverage=LEV)
t0 = time.time()

m1 = binance.load(start="2024-08")
b5 = add_indicators(resample(m1, "5min"), ema_span=300)
b4 = load_4h(data.load_1h())
starts = pd.date_range("2024-10-01", "2026-08-26", freq="D", tz="UTC")
rows5, rows4 = b5.to_dict("records"), b4.to_dict("records")

STRATS = {
    "F1 5 min ruptura 20/10 + tendencia + piramidar": (b5, rows5, lambda: breakout(20, 10, trend_filter=True, pyramid=True, max_leverage=LEV)),
    "F2 5 min ruptura 48/20 + tendencia + piramidar": (b5, rows5, lambda: breakout(48, 20, trend_filter=True, pyramid=True, max_leverage=LEV)),
    "F3 5 min ruptura 20/10 + tendencia, sin piramidar": (b5, rows5, lambda: breakout(20, 10, trend_filter=True)),
    "S4 4 h (la del bot actual)": (b4, rows4, lambda: breakout(trend_filter=True, pyramid=True, max_leverage=LEV)),
}
for seed in range(3):
    STRATS[f"F4 5 min control al azar (semilla {seed})"] = (b5, rows5, lambda s=seed: random_entries(seed=s, prob=0.01))

res = {}
for name, (bars, rows, make) in STRATS.items():
    ends, hits, trades = [], [], []
    for st in starts:
        r = run(bars, make(), st, days=30, capital=CAPITAL, target=TARGET, rows=rows, costs=COSTS)
        ends.append(r.equity_end); hits.append(r.hit_target); trades.append(r.trades)
    ends = np.array(ends)
    res[name] = dict(p_target=np.mean(hits), p_loss=np.mean(ends < CAPITAL), p_ruin=np.mean(ends <= 10),
                     median=np.median(ends), mean=ends.mean(), trades=np.mean(trades))
    print(f"{name}: {100 * np.mean(hits):.1f} % ({time.time() - t0:.0f} s)", flush=True)

ctrl = [v for k, v in res.items() if k.startswith("F4")]
res = {k: v for k, v in res.items() if not k.startswith("F4")}
res["F4 5 min control al azar (media de 3)"] = {k: float(np.mean([c[k] for c in ctrl])) for k in ctrl[0]}

out = ["# 5 minutos con los 2 últimos años (idea de Arturo)\n",
       f"Generado por `research/09_5min_ultimos_2_anos.py` el {datetime.now():%d/%m/%Y}. {len(starts)} ventanas de 30 días "
       f"(inicios del {starts[0]:%d/%m/%Y} al {starts[-1]:%d/%m/%Y}), unos {len(starts) / 30:.0f} meses independientes. 9x, comisión taker 0,05 %.\n",
       "| Estrategia | Llega a 500 € | Acaba perdiendo | Pierde ≥ 90 % | Resultado mediano | Resultado medio | Operaciones/mes |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for name, v in res.items():
    out.append(f"| {name} | {100 * v['p_target']:.1f} % | {100 * v['p_loss']:.0f} % | {100 * v['p_ruin']:.0f} % | "
               f"{v['median']:.0f} € | {v['mean']:.0f} € | {v['trades']:.0f} |")
out.append(f"\nTiempo de cálculo: {time.time() - t0:.0f} s.")
(HERE / "09_5min_ultimos_2_anos.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
