"""Descarga todo lo necesario para el bot BTC 5m. Se puede relanzar: continúa donde lo dejó.

Uso (desde la raíz del repo): python research/05_descargar_btc5m.py [desde YYYY-MM-DD] [hasta YYYY-MM-DD]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.btc5m import binance, polymarket  # noqa: E402

start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-02-15", tz="UTC")
end = pd.Timestamp(sys.argv[2], tz="UTC") if len(sys.argv) > 2 else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)

print("Binance:", len(binance.download("2023-01")), "archivos", flush=True)
days = list(pd.date_range(start, end, freq="D"))
for day in reversed(days):  # primero lo más reciente
    t = time.time()
    for attempt in range(3):
        try:
            n_m, n_t = polymarket.download_day(day)
            break
        except Exception as e:  # noqa: BLE001
            print(f"{day:%Y-%m-%d} reintento {attempt + 1}: {e}", flush=True)
            time.sleep(10)
    else:
        continue
    if n_m >= 0:
        print(f"{day:%Y-%m-%d}: {n_m} mercados, {n_t} operaciones ({time.time() - t:.0f} s)", flush=True)
