"""Bot de alto riesgo: ¿con qué probabilidad se pasa de 100 € a 500 € en 30 días?

Declarado antes de ejecutar. Perpetuo de BTC en Kraken (Kraken permite 10x; se usa 9x; comisiones y funding de sim.py).
Cada estrategia se simula en ventanas de 30 días que empiezan cada día:
- Desarrollo: inicios del 01/07/2019 al 30/11/2025.
- Test intocable: inicios del 01/12/2025 al 26/08/2026 (solo con --holdout).
Estrategias:
  S1 ruptura 20/10 velas de 4 h, stop 2 ATR, riesgo 30 %
  S2 S1 + filtro de tendencia (~50 días)
  S3 S1 + piramidar
  S4 S1 + filtro + piramidar
  S5 todo o nada a favor de la tendencia, 10x, stop 4 %
  S6 control: entradas al azar con el tamaño de S1 (5 semillas)
Uso (desde la raíz): python research/08_bot_arriesgado.py [--holdout]
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.perps import data  # noqa: E402
from phoenix.perps.sim import Costs, run  # noqa: E402
from phoenix.perps.strategies import bold_trend, breakout, load_4h, random_entries  # noqa: E402

HOLDOUT = "--holdout" in sys.argv
HERE = Path(__file__).resolve().parent
CAPITAL, TARGET = 100.0, 500.0
LEV = 9.0  # el bot usa 9x para tener margen al piramidar (Kraken permite 10x)
COSTS = Costs(max_leverage=LEV)

t0 = time.time()
b4 = load_4h(data.load_1h())
rows = b4.to_dict("records")
if HOLDOUT:
    starts = pd.date_range("2025-12-01", "2026-08-26", freq="D", tz="UTC")
else:
    starts = pd.date_range("2019-07-01", "2025-11-30", freq="D", tz="UTC")

STRATS = {
    "S1 Ruptura 20/10, stop 2 ATR, riesgo 30 %": lambda: breakout(),
    "S2 S1 + filtro de tendencia": lambda: breakout(trend_filter=True),
    "S3 S1 + piramidar": lambda: breakout(pyramid=True, max_leverage=LEV),
    "S4 S1 + filtro + piramidar": lambda: breakout(trend_filter=True, pyramid=True, max_leverage=LEV),
    "S5 Todo o nada a favor de la tendencia (9x, stop 4 %)": lambda: bold_trend(lev=LEV),
}
for seed in range(5):
    STRATS[f"S6 Control al azar (semilla {seed})"] = (lambda s=seed: random_entries(seed=s))

results = {}
for name, make in STRATS.items():
    ends, hits, ruins, trades = [], [], [], []
    for st in starts:
        r = run(b4, make(), st, days=30, capital=CAPITAL, target=TARGET, rows=rows, costs=COSTS)
        ends.append(r.equity_end); hits.append(r.hit_target); ruins.append(r.equity_end <= 0.1 * CAPITAL); trades.append(r.trades)
    ends = np.array(ends)
    results[name] = {"p_target": np.mean(hits), "p_ruin": np.mean(ruins), "p_loss": np.mean(ends < CAPITAL),
                     "median": np.median(ends), "mean": ends.mean(), "trades": np.mean(trades),
                     "by_year": pd.Series(hits, index=starts).groupby(starts.year).mean()}
    print(f"{name}: P(500 €)={100 * np.mean(hits):.1f} %  media={ends.mean():.0f} €  ({time.time() - t0:.0f} s)", flush=True)

ctrl = [v for k, v in results.items() if k.startswith("S6")]
results = {k: v for k, v in results.items() if not k.startswith("S6")}
results["S6 Control al azar (media de 5 semillas)"] = {
    "p_target": np.mean([c["p_target"] for c in ctrl]), "p_ruin": np.mean([c["p_ruin"] for c in ctrl]),
    "p_loss": np.mean([c["p_loss"] for c in ctrl]), "median": np.mean([c["median"] for c in ctrl]),
    "mean": np.mean([c["mean"] for c in ctrl]), "trades": np.mean([c["trades"] for c in ctrl]),
    "by_year": sum(c["by_year"] for c in ctrl) / len(ctrl)}

title = "Test intocable" if HOLDOUT else "Desarrollo"
out = [f"# Bot de alto riesgo: de 100 € a 500 € en 30 días ({title})\n",
       f"Generado por `research/08_bot_arriesgado.py` el {datetime.now():%d/%m/%Y}. {len(starts)} ventanas de 30 días "
       f"(inicios del {starts[0]:%d/%m/%Y} al {starts[-1]:%d/%m/%Y}); las ventanas se solapan, así que equivalen a unos "
       f"{len(starts) / 30:.0f} meses independientes. Perpetuo de BTC con 9x como máximo, comisión de taker 0,05 %, funding, liquidación.\n",
       "| Estrategia | Llega a 500 € | Acaba perdiendo | Pierde ≥ 90 % | Resultado mediano | Resultado medio | Operaciones/mes |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for name, v in results.items():
    out.append(f"| {name} | {100 * v['p_target']:.1f} % | {100 * v['p_loss']:.0f} % | {100 * v['p_ruin']:.0f} % | "
               f"{v['median']:.0f} € | {v['mean']:.0f} € | {v['trades']:.1f} |")
out.append("\n## Probabilidad de llegar a 500 € por año de inicio\n")
years = list(next(iter(results.values()))["by_year"].index)
out.append("| Estrategia | " + " | ".join(str(y) for y in years) + " |")
out.append("| --- |" + " --- |" * len(years))
for name, v in results.items():
    out.append(f"| {name.split()[0]} | " + " | ".join(f"{100 * x:.0f} %" for x in v["by_year"]) + " |")
out.append(f"\nTiempo de cálculo: {time.time() - t0:.0f} s.")
suffix = "_holdout" if HOLDOUT else ""
(HERE / f"08_bot_arriesgado{suffix}.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
