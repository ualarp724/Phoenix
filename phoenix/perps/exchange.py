"""Acceso al perpetuo BTC/USD de Kraken (PF_XBTUSD) con la misma interfaz para real, demo y papel.

- KrakenPerp(mode="demo"): entorno de pruebas de Kraken (demo-futures.kraken.com), dinero ficticio.
- KrakenPerp(mode="live"): cuenta real. Las claves se leen de las variables de entorno
  KRAKEN_FUTURES_KEY y KRAKEN_FUTURES_SECRET (o de un archivo .env que no se sube a git).
  Crea las claves SIN permiso de retirada.
- PaperPerp: cuenta simulada en memoria para el modo "dry" y los tests; usa velas públicas.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SYMBOL = "BTC/USD:USD"
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Pos:
    side: int      # +1 largo, -1 corto
    qty: float     # BTC
    entry: float


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def public_client():
    import ccxt
    return ccxt.krakenfutures({"requests_trust_env": True, "enableRateLimit": True})


def candles_4h(client, n: int = 1200) -> pd.DataFrame:
    """Últimas `n` velas de 4 h CERRADAS del perpetuo."""
    since = int((time.time() - (n + 2) * 4 * 3600) * 1000)
    rows = []
    while True:
        batch = client.fetch_ohlcv(SYMBOL, "4h", since=since, limit=500)
        if not batch:
            break
        rows += batch
        if len(batch) < 2 or batch[-1][0] <= since:
            break
        since = batch[-1][0] + 1
        if batch[-1][0] >= (time.time() - 4 * 3600) * 1000:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(hours=4) <= now].sort_index().astype(float)


class KrakenPerp:
    def __init__(self, mode: str = "demo"):
        import ccxt
        if mode not in ("demo", "live"):
            raise ValueError("mode debe ser 'demo' o 'live'")
        _load_env()
        prefix = "KRAKEN_DEMO_FUTURES" if mode == "demo" else "KRAKEN_FUTURES"
        key, secret = os.environ.get(f"{prefix}_KEY"), os.environ.get(f"{prefix}_SECRET")
        if not key or not secret:
            raise RuntimeError(f"Faltan {prefix}_KEY / {prefix}_SECRET en el entorno o en .env")
        self.ex = ccxt.krakenfutures({"apiKey": key, "secret": secret, "requests_trust_env": True, "enableRateLimit": True})
        if mode == "demo":
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        self.mode = mode

    # --- lectura ---
    def candles(self, n: int = 1200) -> pd.DataFrame:
        return candles_4h(self.ex, n)

    def price(self) -> float:
        return float(self.ex.fetch_ticker(SYMBOL)["last"])

    def equity(self) -> float:
        """Valor de la cuenta de margen (flex) en USD."""
        bal = self.ex.fetch_balance()
        flex = (bal.get("info", {}).get("accounts", {}) or {}).get("flex", {})
        for k in ("portfolioValue", "balanceValue", "marginEquity"):
            if flex.get(k) is not None:
                return float(flex[k])
        return float(bal["total"].get("USD", 0.0))

    def position(self) -> Pos | None:
        for p in self.ex.fetch_positions([SYMBOL]):
            qty = float(p.get("contracts") or 0)
            if qty > 0:
                return Pos(1 if p["side"] == "long" else -1, qty, float(p["entryPrice"]))
        return None

    # --- órdenes ---
    def set_leverage(self, lev: float):
        self.ex.set_leverage(int(lev), SYMBOL)

    def market(self, side: int, qty: float, reduce_only: bool = False) -> float:
        o = self.ex.create_order(SYMBOL, "market", "buy" if side == 1 else "sell", qty, None,
                                 {"reduceOnly": True} if reduce_only else {})
        return float(o.get("average") or o.get("price") or self.price())

    def _cancel_kind(self, kind: str):
        for o in self.ex.fetch_open_orders(SYMBOL):
            is_stop = (o.get("triggerPrice") or o.get("stopPrice")) is not None
            if (kind == "stop" and is_stop) or (kind == "tp" and not is_stop):
                self.ex.cancel_order(o["id"], SYMBOL)

    def set_stop(self, pos: Pos, stop: float):
        self._cancel_kind("stop")
        self.ex.create_order(SYMBOL, "market", "sell" if pos.side == 1 else "buy", pos.qty, None,
                             {"stopLossPrice": round(stop), "triggerSignal": "mark"})

    def set_take_profit(self, pos: Pos, price: float):
        self._cancel_kind("tp")
        self.ex.create_order(SYMBOL, "limit", "sell" if pos.side == 1 else "buy", pos.qty, round(price),
                             {"reduceOnly": True})

    def close_all(self):
        self.ex.cancel_all_orders(SYMBOL)
        pos = self.position()
        if pos:
            self.market(-pos.side, pos.qty, reduce_only=True)


class PaperPerp:
    """Cuenta simulada: órdenes a mercado al precio indicado y stop / take profit evaluados vela a vela."""

    def __init__(self, equity: float, taker: float = 0.0005, client=None):
        self.cash, self.taker, self.client = equity, taker, client
        self.pos: Pos | None = None
        self.stop = self.tp = None
        self.last = None
        self.fills: list[tuple] = []

    def candles(self, n: int = 1200) -> pd.DataFrame:
        return candles_4h(self.client, n)

    def price(self) -> float:
        return self.last

    def equity(self) -> float:
        return self.cash + (self.pos.side * self.pos.qty * (self.last - self.pos.entry) if self.pos else 0.0)

    def position(self) -> Pos | None:
        return self.pos

    def set_leverage(self, lev: float):
        pass

    def market(self, side: int, qty: float, reduce_only: bool = False) -> float:
        px = self.last
        self.cash -= qty * px * self.taker
        if self.pos is None:
            self.pos = Pos(side, qty, px)
        elif side == self.pos.side:
            new = self.pos.qty + qty
            self.pos = Pos(side, new, (self.pos.entry * self.pos.qty + px * qty) / new)
        else:
            closed = min(qty, self.pos.qty)
            self.cash += self.pos.side * closed * (px - self.pos.entry)
            left = self.pos.qty - closed
            self.pos = Pos(self.pos.side, left, self.pos.entry) if left > 1e-12 else None
            if self.pos is None:
                self.stop = self.tp = None
        self.fills.append((side, qty, px, reduce_only))
        return px

    def set_stop(self, pos: Pos, stop: float):
        self.stop = stop

    def set_take_profit(self, pos: Pos, price: float):
        self.tp = price

    def close_all(self):
        if self.pos:
            self.market(-self.pos.side, self.pos.qty, reduce_only=True)
        self.stop = self.tp = None

    def on_price(self, p: float):
        self.on_candle(p, p, p, p)

    def on_candle(self, o: float, h: float, lo: float, c: float):
        """Avanza el mercado una vela: stop primero (peor caso), luego take profit."""
        if self.pos is not None and self.stop is not None:
            s = self.pos.side
            if (lo <= self.stop) if s == 1 else (h >= self.stop):
                self.last = (min(o, self.stop) if s == 1 else max(o, self.stop))
                self.close_all()
            elif self.tp is not None and ((h >= self.tp) if s == 1 else (lo <= self.tp)):
                self.last = self.tp
                self.close_all()
        self.last = c
