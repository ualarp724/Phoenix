"""Fase 1: informe de calidad de datos y de costes de XAUUSD M15.

Del periodo de test solo se miran recuentos e integridad, nunca precios ni resultados.
Uso: python research/01_datos.py  -> escribe research/01_datos.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.costs import max_sl_distance_usd, spread_usd  # noqa: E402
from phoenix.data import dev_data, holdout_data, load_bars, validate_bars  # noqa: E402
from phoenix.settings import DATA_DIR, load_settings  # noqa: E402

S = load_settings()
out = []
w = out.append

bars = load_bars(DATA_DIR / "vantage_gold.csv", S)
rep = validate_bars(bars, 15)
ny_hours = bars.index.tz_convert("America/New_York").hour
dev = dev_data(bars, S)
test = holdout_data(bars, S, i_know_this_is_the_final_test=True)  # solo recuentos

w("# Fase 1 — Datos y costes de XAUUSD M15\n")
w(f"Archivo `data/raw/vantage_gold.csv`, horas en UTC. Generado por `research/01_datos.py`.\n")
w("## Integridad\n")
w("| Comprobación | Resultado |\n| --- | --- |")
w(f"| Velas | {rep.rows:,} ({rep.first:%Y-%m-%d} → {rep.last:%Y-%m-%d}) |")
w(f"| Duplicados / desordenadas / OHLC imposible / precios ≤ 0 | {rep.duplicates} / {rep.non_monotonic} / {rep.bad_ohlc} / {rep.non_positive} |")
w(f"| Velas a las 17:00 de Nueva York (pausa diaria; debe ser 0 si la conversión horaria es correcta) | {(ny_hours == 17).sum()} |")
w(f"| Huecos no explicados por pausa diaria o fin de semana | {len(rep.unexpected_gaps)} |")
w(f"| Velas en desarrollo / test | {len(dev):,} / {len(test):,} |\n")
if len(rep.unexpected_gaps):
    g = rep.unexpected_gaps.sort_values(ascending=False).head(10)
    w("Los 10 huecos más largos (casi todos festivos):\n")
    w("| Reanuda (UTC) | Duración |\n| --- | --- |")
    for t, d in g.items():
        w(f"| {t:%Y-%m-%d %H:%M} | {d} |")
    w("")

# Volatilidad y spread por año (solo desarrollo)
tr = pd.concat([dev.high - dev.low, (dev.high - dev.close.shift()).abs(), (dev.low - dev.close.shift()).abs()], axis=1).max(axis=1)
atr = tr.rolling(14).mean()
risk = S.account.max_risk_usd
max_sl = max_sl_distance_usd(risk, S.instrument)
sp = dev.spread_points.apply(lambda p: spread_usd(p, S.instrument, S.costs))
w("## Volatilidad y spread por año (solo desarrollo)\n")
w(f"Riesgo máximo por operación: {S.account.max_risk_per_trade_pct} % de {S.account.capital_usd:.0f} $ = {risk:.2f} $ → SL máximo con 0,01 lotes = {max_sl:.2f} $/oz.\n")
w("| Año | Precio mediano ($) | ATR14 M15 mediano ($/oz) | Spread vela mediano (pts) | Spread usado ($/oz) | SL máx / ATR |\n| --- | --- | --- | --- | --- | --- |")
for y, grp in dev.groupby(dev.index.year):
    a = atr.loc[grp.index].median()
    w(f"| {y} | {grp.close.median():,.0f} | {a:.2f} | {grp.spread_points.median():.0f} | {sp.loc[grp.index].median():.2f} | {max_sl / a:.2f} |")
w("")

# Coste de una operación típica con 0,01 lotes
c = S.costs
cost = 0.12 * 1 + 2 * c.commission_per_lot_per_side_usd * 0.01 + 2 * c.slippage_per_market_fill_usd * 1
w("## Coste de una operación con 0,01 lotes\n")
w(f"Spread {0.12:.2f} $ + comisión {2*c.commission_per_lot_per_side_usd*0.01:.2f} $ + deslizamiento {2*c.slippage_per_market_fill_usd:.2f} $ (2 ejecuciones a mercado) = **{cost:.2f} $**, un {100*cost/risk:.1f} % del riesgo máximo por operación.\n")

Path(__file__).with_suffix(".md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
