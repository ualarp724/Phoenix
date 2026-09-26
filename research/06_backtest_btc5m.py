"""Backtest del bot BTC 5m con precios REALES de Polymarket (configuraciones declaradas antes de ejecutar).

- Modelo: LightGBM entrenado con los 12 meses anteriores de velas de Binance (objetivo aproximado
  que coincide con el resultado real en el 94 % de los mercados), reentrenado por trimestre.
- Desarrollo: mercados del 15/02/2026 al 28/08/2026. Test intocable: 29/08/2026 en adelante
  (solo con --holdout, una vez, al final).
- Configuraciones:
  C1 decide en el inicio (vela s-1), compra entre +1 y +5 s, margen 0,02
  C2 decide en el inicio, compra entre +1 y +10 s, margen 0,02
  C3 decide en el inicio, compra entre +1 y +5 s, margen 0,05
  C4 decide un minuto antes (vela s-2), compra entre -55 y -5 s, margen 0,02
Uso (desde la raíz): python research/06_backtest_btc5m.py [--holdout]
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.btc5m import binance, polymarket  # noqa: E402
from phoenix.btc5m.backtest import FILL_WINDOWS, fill_summary, prices_from_summary, simulate, summary  # noqa: E402
from phoenix.btc5m.features import features_1m, features_at, proxy_target, window_starts  # noqa: E402
from phoenix.walkforward import LGBM_PARAMS  # noqa: E402

HOLDOUT = "--holdout" in sys.argv
DEV_START, HOLDOUT_START = pd.Timestamp("2026-02-15", tz="UTC"), pd.Timestamp("2026-08-29", tz="UTC")
HERE = Path(__file__).resolve().parent
CONFIGS = {
    "C1 inicio, +1..+5 s, margen 0,02": dict(lag=1, fill=(1, 5), margin=0.02),
    "C2 inicio, +1..+10 s, margen 0,02": dict(lag=1, fill=(1, 10), margin=0.02),
    "C3 inicio, +1..+5 s, margen 0,05": dict(lag=1, fill=(1, 5), margin=0.05),
    "C4 1 min antes, -55..-5 s, margen 0,02": dict(lag=2, fill=(-55, -5), margin=0.02),
}

t0 = time.time()
m = binance.load(start="2024-12")
f1 = features_1m(m)
starts = window_starts(m)
y = proxy_target(m, starts)

if HOLDOUT:
    periods = [(HOLDOUT_START, pd.Timestamp.now(tz="UTC").normalize())]
else:
    periods = [(DEV_START, pd.Timestamp("2026-05-15", tz="UTC")), (pd.Timestamp("2026-05-15", tz="UTC"), pd.Timestamp("2026-08-15", tz="UTC")),
               (pd.Timestamp("2026-08-15", tz="UTC"), HOLDOUT_START)]

# Resumen de precios de compra por mercado (se construye una vez desde las operaciones crudas)
SUMMARY = polymarket.DATA_DIR.parent / "polymarket_btc5m_resumen.csv.gz"
if not SUMMARY.exists():
    mk_all = polymarket.load("markets").dropna(subset=["outcome_up"])
    tr_all = polymarket.load_buys_near_start(rel_from=min(a for a, _ in FILL_WINDOWS), rel_to=max(b for _, b in FILL_WINDOWS))
    fill_summary(tr_all, mk_all).to_csv(SUMMARY, index=False)
    del tr_all
fills = pd.read_csv(SUMMARY)
mk = fills[(fills.start_ts >= periods[0][0].timestamp()) & (fills.start_ts < periods[-1][1].timestamp())].dropna(subset=["outcome_up"])
mk = mk.assign(outcome_up=mk.outcome_up.astype(bool))
mk_time = pd.to_datetime(mk.start_ts, unit="s", utc=True)

# Predicciones fuera de muestra para cada lag
preds = {}
for lag in sorted({c["lag"] for c in CONFIGS.values()}):
    X_all = features_at(m, starts, lag=lag, f1=f1)
    p = pd.Series(np.nan, index=mk.slug.values)
    for a, b in periods:
        tr_mask = (starts >= a - pd.DateOffset(months=12)) & (starts < a - pd.Timedelta(minutes=10))
        Xt, yt = X_all[tr_mask], y[tr_mask]
        ok = Xt.notna().all(axis=1) & yt.notna()
        model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(Xt[ok], yt[ok].astype(int))
        sel = (mk_time >= a) & (mk_time < b)
        Xm = X_all.reindex(mk_time[sel])
        okm = Xm.notna().all(axis=1).to_numpy()
        p.loc[mk.slug[sel].values[okm]] = model.predict_proba(Xm[okm])[:, 1]
    preds[lag] = p

results, bets_by = {}, {}
for name, c in CONFIGS.items():
    px = prices_from_summary(mk, c["fill"])
    bets = simulate(mk, preds[c["lag"]], px, margin=c["margin"])
    results[name], bets_by[name] = summary(bets), bets

# Diagnósticos
y_real = mk.set_index("slug")["outcome_up"].astype(float)
proxy_agree = (y.reindex(mk_time).to_numpy() == y_real.to_numpy())
px1 = prices_from_summary(mk, (1, 5))
p1 = preds[1].reindex(px1.index)
calib = pd.DataFrame({"p": p1, "y": y_real.reindex(px1.index), "px_up": px1["px_up"]}).dropna()
calib["bucket"] = pd.cut(calib.p, [0, .2, .35, .45, .55, .65, .8, 1])
cal = calib.groupby("bucket", observed=True).agg(n=("y", "size"), p_modelo=("p", "mean"), frec_real=("y", "mean"), precio_up=("px_up", "mean"))

title = "Test intocable" if HOLDOUT else "Desarrollo"
out = [f"# Bot BTC 5m — backtest con precios reales de Polymarket ({title})\n",
       f"Generado por `research/06_backtest_btc5m.py` el {datetime.now():%d/%m/%Y}. {len(mk):,} mercados resueltos "
       f"({periods[0][0]:%d/%m/%Y} → {periods[-1][1] - pd.Timedelta(days=1):%d/%m/%Y}). Apuesta fija de 5 $, comisión 0,07·p·(1−p). "
       f"El objetivo de Binance coincide con el resultado real en el {100 * np.nanmean(proxy_agree):.1f} % de estos mercados.\n",
       "| Configuración | Apuestas | Por día | Acierto | Precio medio | Ventaja esperada | Ganancia (5 $/apuesta) | Rentabilidad por apuesta | Días en positivo | Peor caída |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for name, s in results.items():
    if not s.get("bets"):
        out.append(f"| {name} | 0 | | | | | | | | |")
        continue
    out.append(f"| {name} | {s['bets']:,} | {s['bets_per_day']:.0f} | {s['win_rate_pct']:.1f} % | {s['avg_price']:.3f} | "
               f"{s['avg_expected_edge']:+.3f} | {s['pnl_usd']:+,.0f} $ | {s['roi_per_bet_pct']:+.1f} % | {s['days_positive_pct']:.0f} % | {s['max_dd_usd']:,.0f} $ |")
out.append("\n## Calibración (decisión en el inicio, precio de +1..+5 s)\n")
out.append("¿Acierta el modelo lo que dice? ¿Y el mercado ya lo sabía? Si el precio de «Up» sigue a la frecuencia real, el mercado ya lo descuenta.\n")
out.append("| Prob. del modelo | Mercados | Media modelo | Frecuencia real de «Up» | Precio medio de «Up» |\n| --- | --- | --- | --- | --- |")
for b, r in cal.iterrows():
    out.append(f"| {b} | {int(r.n):,} | {r.p_modelo:.3f} | {r.frec_real:.3f} | {r.precio_up:.3f} |")
out.append("\n## Resultado por mes (C1)\n")
b1 = bets_by[list(CONFIGS)[0]]
if len(b1):
    mon = b1.groupby(b1.time.dt.strftime("%Y-%m")).agg(apuestas=("pnl", "size"), acierto=("won", "mean"), ganancia=("pnl", "sum"))
    out.append("| Mes | Apuestas | Acierto | Ganancia |\n| --- | --- | --- | --- |")
    out += [f"| {i} | {int(r.apuestas)} | {100 * r.acierto:.1f} % | {r.ganancia:+.0f} $ |" for i, r in mon.iterrows()]
out.append(f"\nTiempo de cálculo: {time.time() - t0:.0f} s.")
suffix = "_holdout" if HOLDOUT else ""
(HERE / f"06_backtest_btc5m{suffix}.md").write_text("\n".join(out), encoding="utf-8")
for name, b in bets_by.items():
    b.to_csv(HERE / f"06_bets{suffix}_{name.split()[0]}.csv.gz", index=False)
print("\n".join(out))
