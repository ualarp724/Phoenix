"""Fase 3, segunda tanda (declarada antes de ejecutarla).

Hallazgo de la primera tanda: LightGBM predice si el precio SE MOVERÁ (AUC ~0,68)
pero no HACIA DÓNDE (AUC de dirección ~0,51). Hipótesis: combinar una entrada que
elige la dirección por sí misma (ruptura) con el modelo de movimiento como filtro.

Configuraciones (5): B1 Londres TP 2x/32 velas · B2 Londres + tendencia H4 ·
B3 rango de apertura de Nueva York · B4 Donchian filtrada por movimiento (p ≥ p70 de train) ·
B5 Londres filtrada por movimiento (p ≥ p50 de train).
"""
from datetime import datetime




from common import HERE, S, bars, evaluate, feats, log, stats, table, walkforward_proba, windows
from phoenix.labels import tp_first_labels
from phoenix.strategies import donchian_orders, range_breakout_orders

results = {}


def run(name, cfg, orders):
    tr, sk = evaluate(orders)
    results[name] = stats(tr, sk)
    log(name, cfg, results[name])


run("B1 Londres, TP 2x, 32 velas", {"sl_k": 1, "tp_m": 2.0, "max_bars": 32},
    range_breakout_orders(bars, feats, tp_m=2.0, max_bars=32))
run("B2 Londres + tendencia H4", {"sl_k": 1, "tp_m": 1.5, "max_bars": 16, "trend_filter": True},
    range_breakout_orders(bars, feats, trend_filter=True))
run("B3 Rango de apertura de Nueva York (9:30-10:30)", {"sl_k": 1, "tp_m": 1.5, "max_bars": 16, "tz": "NY"},
    range_breakout_orders(bars, feats, tz="America/New_York", range_start=9.5, range_end=10.5, trade_end=13.0))

lab = tp_first_labels(bars, feats["atr"], 1.5 * feats["atr"], 16, S)
move = ((lab["long"] == 1) | (lab["short"] == 1)).astype(float).where(lab["long"].notna())
_, move_pct = walkforward_proba(move)

don = donchian_orders(bars, feats)
don.loc[~(move_pct >= 0.70), "side"] = 0
run("B4 Donchian + filtro de movimiento (p70)", {"sl_k": 1, "tp_m": 1.5, "max_bars": 16, "move_pct": 0.70}, don)

ldn = range_breakout_orders(bars, feats)
ldn.loc[~(move_pct >= 0.50), "side"] = 0
run("B5 Londres + filtro de movimiento (p50)", {"sl_k": 1, "tp_m": 1.5, "max_bars": 16, "move_pct": 0.50}, ldn)

out = ["# Fase 3 — Segunda tanda: rupturas y modelo de movimiento\n",
       f"Generado por `research/03_ruptura_y_movimiento.py` el {datetime.now():%d/%m/%Y}. Mismas {len(windows)} ventanas y reglas que la primera tanda.\n"]
out += table(results)
out.append("\n## R medio por ventana\n")
out.append("| Estrategia | " + " | ".join(f"{w.val_start:%m/%y}" for w in windows) + " |")
out.append("| --- |" + " --- |" * len(windows))
for name, st in results.items():
    if st.get("trades", 0):
        out.append(f"| {name} | " + " | ".join(f"{x:+.2f}" for x in st["per_window_avg_r"]) + " |")
(HERE / "03_ruptura_y_movimiento.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
