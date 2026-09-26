"""¿Hay ventaja en el retraso del precio de referencia de Polymarket (Chainlink) frente al precio real?

Comprobado con Coinbase (BTC-USD): price_to_beat ≈ precio al instante del inicio (cierre de la vela s-1)
y final_price ≈ precio al final, con ~4 $ de diferencia típica. Cuando Coinbase y price_to_beat
difieren mucho al empezar, el resultado tiende a seguir a Coinbase. Pero price_to_beat no se publica
hasta después, así que solo sirve si se tiene el feed de Chainlink en tiempo real y se reacciona en
1-2 segundos. Aquí se mide cuánto quedaba de esa ventaja a los precios reales pagados.
Uso (desde la raíz): python research/07_retraso_chainlink.py -> research/07_retraso_chainlink.md
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.btc5m import coinbase, polymarket  # noqa: E402
from phoenix.btc5m.backtest import fee_per_share, prices_from_summary  # noqa: E402

HERE = Path(__file__).resolve().parent
fills = pd.read_csv(polymarket.DATA_DIR.parent / "polymarket_btc5m_resumen.csv.gz")
fills = fills[fills.start_ts < pd.Timestamp("2026-08-29", tz="UTC").timestamp()].dropna(subset=["outcome_up", "price_to_beat"])
cb = coinbase.load()
s = pd.to_datetime(fills.start_ts, unit="s", utc=True)
gap = cb["close"].reindex(s - pd.Timedelta(minutes=1)).to_numpy() - fills["price_to_beat"].to_numpy()
y = fills["outcome_up"].astype(bool).to_numpy()
side_up = gap > 0
won = np.where(side_up, y, ~y)

out = ["# Retraso de Chainlink: ¿se puede aprovechar?\n",
       f"Generado por `research/07_retraso_chainlink.py` el {datetime.now():%d/%m/%Y}. {len(fills):,} mercados (15/02 – 28/08/2026). "
       "Estrategia: apostar al lado que indica Coinbase frente a price_to_beat cuando la diferencia es grande. "
       "Rentabilidad por dólar apostado, con comisión, al precio medio pagado por los takers en cada ventana.\n",
       "| Diferencia | Mercados | Acierto | +0..+2 s | +1..+5 s | +5..+15 s |", "| --- | --- | --- | --- | --- | --- |"]
for q in (0.8, 0.9, 0.95):
    thr = np.nanquantile(np.abs(gap), q)
    sel = (np.abs(gap) >= thr) & ~np.isnan(gap)
    cells = []
    for w in [(0, 2), (1, 5), (5, 15)]:
        px = prices_from_summary(fills, w)
        price = np.where(side_up, px["px_up"].to_numpy(), px["px_down"].to_numpy())
        k = sel & ~np.isnan(price)
        roi = (won[k] - price[k] - fee_per_share(price[k])) / price[k]
        se = roi.std() / np.sqrt(k.sum())
        cells.append(f"{100 * roi.mean():+.1f} % (± {196 * se:.1f})")
    out.append(f"| ≥ {thr:.0f} $ (el {100 * (1 - q):.0f} % mayor) | {sel.sum():,} | {100 * won[sel].mean():.1f} % | " + " | ".join(cells) + " |")
out.append("\nLa ventaja solo aparece en los dos primeros segundos y se evapora en cuanto pasan unos pocos: es una carrera de "
           "latencia contra bots que ven el feed de Chainlink en tiempo real. price_to_beat no se publica durante la ventana.")
(HERE / "07_retraso_chainlink.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
