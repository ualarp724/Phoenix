"""Bot de alto riesgo sobre el perpetuo de BTC en Kraken: estrategia S4 de research/08.

Ruptura del canal de 20 velas de 4 h a favor de la tendencia de ~50 días, stop de 2 ATR,
30 % del capital en riesgo por operación, apalancamiento máximo 10x y piramidar cada +1 ATR.
Se para solo al llegar al objetivo (5x el capital inicial), si el capital baja del mínimo
o a los 30 días. El stop y el objetivo quedan puestos en el exchange: si el ordenador se
apaga, la posición sigue protegida.

Uso:
  python -m phoenix.perps.bot --mode dry            # simulación con velas reales, sin claves
  python -m phoenix.perps.bot --mode demo --check   # comprueba claves de la cuenta demo de Kraken
  python -m phoenix.perps.bot --mode demo           # opera en la cuenta demo (dinero ficticio)
  python -m phoenix.perps.bot --mode live --i-accept-losing-everything   # dinero real
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from phoenix.perps.exchange import KrakenPerp, PaperPerp, Pos, public_client
from phoenix.perps.sim import Plan
from phoenix.perps.strategies import breakout, load_4h

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOT = 0.0001  # BTC, tamaño mínimo y paso del perpetuo de Kraken


@dataclass
class Config:
    target_multiple: float = 5.0
    days: int = 30
    max_leverage: float = 9.0  # Kraken permite 10x; se deja margen para poder piramidar sin rechazos
    min_equity: float = 5.0
    max_start_equity: float = 150.0  # protección: no arrancar si la cuenta tiene más de lo previsto


@dataclass
class State:
    started: str | None = None
    start_equity: float | None = None
    last_candle: str | None = None
    stop: float | None = None
    memory: dict = field(default_factory=dict)  # precio de la última compra para piramidar
    finished: str | None = None

    @classmethod
    def load(cls, path: Path) -> "State":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))


class Bot:
    def __init__(self, exchange, config: Config, state_path: Path, log: logging.Logger):
        self.x, self.cfg, self.state_path, self.log = exchange, config, state_path, log
        self.st = State.load(state_path)
        self.strategy = breakout(trend_filter=True, pyramid=True, max_leverage=config.max_leverage, memory=self.st.memory)

    # --- utilidades ---
    def _qty(self, equity: float, lev: float, price: float) -> float:
        return math.floor(equity * min(lev, self.cfg.max_leverage) / price / LOT) * LOT

    def _target(self) -> float:
        return self.st.start_equity * self.cfg.target_multiple

    def _tp_price(self, pos: Pos, equity: float, price: float) -> float:
        return price + pos.side * (self._target() - equity) / pos.qty

    def _finish(self, why: str):
        self.x.close_all()
        self.st.finished = f"{pd.Timestamp.now(tz='UTC').isoformat()} {why}"
        self.st.save(self.state_path)
        self.log.info("FIN: %s. Capital final %.2f $", why, self.x.equity())

    # --- un paso del bucle ---
    def step(self, candles: pd.DataFrame, now: pd.Timestamp | None = None) -> bool:
        """Devuelve False cuando el bot ha terminado."""
        now = now or pd.Timestamp.now(tz="UTC")
        if self.st.finished:
            return False
        equity = self.x.equity()
        if self.st.started is None:
            if equity > self.cfg.max_start_equity:
                raise RuntimeError(f"La cuenta tiene {equity:.2f} $, más de {self.cfg.max_start_equity} $: "
                                   "deja solo el dinero que quieres arriesgar o sube max_start_equity.")
            self.st.started, self.st.start_equity = now.isoformat(), equity
            self.x.set_leverage(self.cfg.max_leverage)
            self.log.info("Inicio con %.2f $. Objetivo %.2f $ en %d días.", equity, self._target(), self.cfg.days)
            self.st.save(self.state_path)

        if equity >= self._target():
            self._finish("objetivo alcanzado")
            return False
        if equity <= self.cfg.min_equity:
            self._finish("capital por debajo del mínimo")
            return False
        if now >= pd.Timestamp(self.st.started) + pd.Timedelta(days=self.cfg.days):
            self._finish("fin del plazo")
            return False

        last = candles.index[-1]
        if self.st.last_candle == last.isoformat():
            return True  # nada nuevo: las órdenes en el exchange hacen el resto
        self.st.last_candle = last.isoformat()

        b4 = load_4h(candles)
        row = b4.iloc[-1].to_dict()
        pos = self.x.position()
        if pos is None and self.st.stop is not None:
            self.log.info("Posición cerrada por stop u objetivo. Capital %.2f $", equity)
            self.st.stop = None
        view = None if pos is None else Pos(pos.side, pos.qty, pos.entry)
        if view is not None:
            view.stop = self.st.stop if self.st.stop is not None else pos.entry * (1 - pos.side * 0.04)
        plan: Plan | None = self.strategy({"i": len(b4) - 1, "row": row, "pos": view, "equity": equity})
        price = self.x.price() or row["close"]
        if plan is not None:
            self._execute(plan, pos, equity, price)
        self.st.save(self.state_path)
        return True

    def _execute(self, plan: Plan, pos: Pos | None, equity: float, price: float):
        if pos is not None and plan.side != pos.side:
            self.x.close_all()
            pos = None
        if plan.side == 0:
            return
        if pos is None:
            qty = self._qty(equity, plan.leverage, price)
            if qty < LOT:
                self.log.info("Señal %s pero el tamaño es menor que el mínimo", plan.side)
                return
            fill = self.x.market(plan.side, qty)
            pos = self.x.position() or Pos(plan.side, qty, fill)
            self.log.info("ABRE %s %.4f BTC a %.0f, stop %.0f (%.1fx)", "LARGO" if plan.side == 1 else "CORTO",
                          qty, fill, plan.stop, qty * fill / equity)
        elif plan.leverage > 0:  # piramidar
            add = self._qty(equity, plan.leverage, price) - pos.qty
            if add >= LOT:
                self.x.market(pos.side, add)
                pos = self.x.position() or Pos(pos.side, pos.qty + add, pos.entry)
                self.log.info("PIRAMIDA +%.4f BTC a %.0f (total %.4f)", add, price, pos.qty)
        self.st.stop = plan.stop
        self.x.set_stop(pos, plan.stop)
        self.x.set_take_profit(pos, self._tp_price(pos, self.x.equity(), price))
        self.log.info("Stop en %.0f, objetivo en %.0f", plan.stop, self._tp_price(pos, self.x.equity(), price))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["dry", "demo", "live"], default="dry")
    ap.add_argument("--check", action="store_true", help="solo comprobar conexión, saldo y posición")
    ap.add_argument("--i-accept-losing-everything", action="store_true")
    ap.add_argument("--dry-equity", type=float, default=114.0, help="capital simulado en modo dry (USD)")
    a = ap.parse_args()
    if a.mode == "live" and not a.i_accept_losing_everything:
        raise SystemExit("Para operar con dinero real añade --i-accept-losing-everything")

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / f"perps_bot_{a.mode}.log"), logging.StreamHandler()])
    log = logging.getLogger("perps")
    x = PaperPerp(a.dry_equity, client=public_client()) if a.mode == "dry" else KrakenPerp(a.mode)
    if a.check:
        c = x.candles(50)
        log.info("Conexión OK. Última vela cerrada %s, cierre %.0f", c.index[-1], c["close"].iloc[-1])
        if a.mode != "dry":
            log.info("Capital %.2f $, posición %s", x.equity(), x.position())
        return
    bot = Bot(x, Config(), LOG_DIR / f"perps_state_{a.mode}.json", log)
    while True:
        try:
            candles = x.candles()
            if a.mode == "dry":
                x.on_price(float(public_client().fetch_ticker("BTC/USD:USD")["last"]))
            if not bot.step(candles):
                break
        except Exception as e:  # noqa: BLE001 — un fallo de red no debe tumbar el bot
            log.exception("Error en el bucle: %s", e)
        time.sleep(60)


if __name__ == "__main__":
    main()
