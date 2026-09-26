"""¿Se puede predecir si BTC sube o baja en los próximos 5 minutos? (idea Polymarket "Up or Down 5m")

Estudio declarado antes de ejecutarlo:
- Datos: velas de 1 minuto BTCUSDT de Binance (data.binance.vision), ene 2023 – ago 2026.
- Pregunta: al empezar cada ventana de 5 min (:00, :05...), ¿acabará "Up"? Polymarket resuelve con el
  TWAP de Chainlink (streams de 60 s), no con el cierre, así que se prueban tres aproximaciones:
    cierre: cierre del minuto 5 >= apertura de la ventana
    twap_ventana: media de los 5 minutos >= precio medio del minuto anterior al inicio
    twap_final: precio medio del minuto 5 >= precio medio del minuto anterior al inicio
  Solo se usan velas ya cerradas antes del inicio de la ventana.
- Test intocable: jun–ago 2026 (no se evalúa aquí).
- Walk-forward: 12 meses de entrenamiento, 3 de validación, avanzando 3 meses.
- Umbral de rentabilidad: comprar a 0,505 (0,50 + medio céntimo de horquilla) y pagar la comisión
  de taker de Polymarket en cripto (0,07 · p · (1-p) ≈ 0,0175 por acción) → hace falta acertar > 52,25 %.
  Es una cota optimista: supone que el mercado siempre cotiza 50/50 al empezar.
Uso: python research/04_btc_5min.py -> research/04_btc_5min.md
"""
import glob
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.walkforward import LGBM_PARAMS, make_windows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "binance_btcusdt_1m"
HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
ENTRY_PRICE = 0.505
FEE = 0.07 * ENTRY_PRICE * (1 - ENTRY_PRICE)
BREAKEVEN = ENTRY_PRICE + FEE
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"]


def load_1m() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(DATA / "BTCUSDT-1m-*.zip"))):
        with zipfile.ZipFile(f) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
        if not str(df.iloc[0, 0]).isdigit():  # algunos meses traen cabecera
            df = df.iloc[1:].astype({"open_time": "int64"})
        ts = df["open_time"].astype("int64")
        unit = "us" if ts.iloc[0] > 1e14 else "ms"  # Binance pasó a microsegundos en 2025
        df.index = pd.to_datetime(ts, unit=unit, utc=True)
        parts.append(df[["open", "high", "low", "close", "volume", "taker_buy_base"]].astype(float))
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated()]


def features_1m(m: pd.DataFrame) -> pd.DataFrame:
    """Features en la fila t usando solo velas de 1 min hasta t (incluida)."""
    c = m["close"]
    lr = np.log(c).diff()
    vol60 = lr.rolling(60).std()
    f = pd.DataFrame(index=m.index)
    for n in (1, 5, 15, 60, 240):
        f[f"r{n}"] = np.log(c / c.shift(n)) / (vol60 * np.sqrt(n))
    f["vol60"] = vol60
    f["vol_ratio"] = vol60 / lr.rolling(1440).std()
    v = m["volume"]
    f["taker5"] = m["taker_buy_base"].rolling(5).sum() / v.rolling(5).sum()
    f["taker60"] = m["taker_buy_base"].rolling(60).sum() / v.rolling(60).sum()
    f["vol_rel5"] = v.rolling(5).sum() / (v.rolling(1440).mean() * 5)
    lo, hi = m["low"].rolling(60).min(), m["high"].rolling(60).max()
    f["pos60"] = (c - lo) / (hi - lo)
    o5, h5, l5 = m["open"].shift(4), m["high"].rolling(5).max(), m["low"].rolling(5).min()
    rng5 = (h5 - l5).replace(0, np.nan)
    f["body5"] = (c - o5) / rng5
    f["upwick5"] = (h5 - np.maximum(o5, c)) / rng5
    hour = m.index.hour + m.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = m.index.dayofweek
    return f


def main():
    m = load_1m()
    f1 = features_1m(m)
    starts = m.index[(m.index.minute % 5 == 0)]
    starts = starts[starts + pd.Timedelta(minutes=4) <= m.index[-1]]
    # Features al inicio de la ventana = fila del minuto anterior (ya cerrado)
    X = f1.shift(1).reindex(starts)
    typ = (m["high"] + m["low"] + m["close"]) / 3  # precio medio aproximado de cada minuto
    def at(series, minutes):
        return series.reindex(starts + pd.Timedelta(minutes=minutes)).to_numpy()
    ref = at(typ, -1)
    targets = {
        "cierre": (at(m["close"], 4), at(m["open"], 0)),
        "twap_ventana": (np.nanmean(np.vstack([at(typ, k) for k in range(5)]), axis=0), ref),
        "twap_final": (at(typ, 4), ref),
    }
    ys = {}
    for name, (end, start) in targets.items():
        yy = pd.Series((end >= start).astype(float), index=starts)
        yy[np.isnan(end) | np.isnan(start)] = np.nan
        ys[name] = yy
    y = ys["cierre"]
    ok = X.notna().all(axis=1) & pd.concat(ys, axis=1).notna().all(axis=1) & (X.index < HOLDOUT_START)
    X = X[ok]
    ys = {k: v[ok] for k, v in ys.items()}

    windows = make_windows(X.index[0], HOLDOUT_START, train_months=12, val_months=3)

    def walk(y):
        preds, imps = [], []
        for w in windows:
            tr = (X.index >= w.train_start) & (X.index < w.train_end - pd.Timedelta(minutes=5))
            va = (X.index >= w.val_start) & (X.index < w.val_end)
            model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(X[tr], y[tr].astype(int))
            conf_train = np.sort(np.abs(model.predict_proba(X[tr])[:, 1] - 0.5))
            p = model.predict_proba(X[va])[:, 1]
            preds.append(pd.DataFrame({"p": p, "y": y[va].to_numpy(),
                                       "conf_pct": np.searchsorted(conf_train, np.abs(p - 0.5)) / len(conf_train),
                                       "mom": np.sign(X.loc[va, "r5"].to_numpy()), "window": w.val_start}, index=X.index[va]))
            imps.append(pd.Series(model.booster_.feature_importance("gain"), index=X.columns))
        P = pd.concat(preds)
        P["hit"] = ((P.p >= 0.5) == (P.y == 1)).astype(float)
        P["mom_hit"] = ((P.mom >= 0) == (P.y == 1)).astype(float)
        imp = pd.concat(imps, axis=1).mean(axis=1)
        return P, (imp / imp.sum()).sort_values(ascending=False)

    days = X.index[X.index >= windows[0].val_start].normalize().nunique()

    def row(name, hits):
        n, acc = len(hits), hits.mean()
        ci = 1.96 * np.sqrt(acc * (1 - acc) / n) if n else np.nan
        return f"| {name} | {n:,} | {n / days:.0f} | {100 * acc:.2f} % (± {100 * ci:.2f}) | {100 * (acc - BREAKEVEN):+.2f} c |"

    out = ["# ¿Se puede predecir BTC a 5 minutos? (idea Polymarket)\n",
           f"Generado por `research/04_btc_5min.py` el {datetime.now():%d/%m/%Y}. Velas de 1 min de Binance, "
           f"{len(windows)} ventanas de validación de 3 meses ({windows[0].val_start:%m/%Y} → {HOLDOUT_START - pd.Timedelta(days=1):%m/%Y}). "
           f"Para ganar dinero hay que acertar más del **{100 * BREAKEVEN:.2f} %** (entrada a {ENTRY_PRICE} + comisión de {FEE:.4f}), "
           "suponiendo que se puede comprar a 0,505 justo al empezar.\n"]
    for tname, y in ys.items():
        P, imp = walk(y)
        out += [f"\n## Objetivo «{tname}»\n",
                "| Estrategia | Apuestas | Por día | Acierto (IC 95 %) | Ganancia media por acción |",
                "| --- | --- | --- | --- | --- |",
                row("Siempre «sube»", P.y), row("Momentum (repetir los últimos 5 min)", P.mom_hit), row("LightGBM, todas", P.hit)]
        for q in (0.8, 0.9, 0.95, 0.99):
            out.append(row(f"LightGBM, solo el {100 * (1 - q):.0f} % más seguro", P.hit[P.conf_pct >= q]))
        per_w = P[P.conf_pct >= 0.95].groupby("window")["hit"].mean()
        out.append(f"\nAcierto del 5 % más seguro por ventana: " + ", ".join(f"{100 * v:.1f} %" for v in per_w)
                   + f". Features con más peso: " + ", ".join(f"`{k}` {100 * v:.0f} %" for k, v in imp.head(5).items()) + ".")
    (Path(__file__).with_suffix(".md")).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
