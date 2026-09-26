import pandas as pd
import pytest

from phoenix.perps.sim import Costs, Plan, liquidation_price, run

C = Costs(taker=0.0005, slippage=0.0, funding_long_per_hour=0.0)


def _bars(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _once(plan):
    done = {"x": False}

    def strat(st):
        if not done["x"]:
            done["x"] = True
            return plan
        return None
    return strat


def test_liquidation_price_10x():
    assert liquidation_price(100.0, 1, 10, C) == pytest.approx(95.0)
    assert liquidation_price(100.0, -1, 10, C) == pytest.approx(105.0)


def test_long_reaches_target_and_stops():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 150, 99, 140], [140, 140, 140, 140]])
    r = run(bars, _once(Plan(1, 97.0, 10.0)), bars.index[0], days=1, capital=100, target=150)
    assert r.hit_target and r.equity_end == pytest.approx(150, abs=0.01)


def test_stop_loss_costs():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 100, 96, 97], [97, 97, 97, 97]])
    r = run(bars, _once(Plan(1, 97.0, 10.0)), bars.index[0], days=1, capital=100, costs=C)
    # 10 BTC-equivalente de 100 $ -> 10x: pierde 3 % * 1000 = 30 $ + comisiones (0,5 + 0,485)
    assert r.equity_end == pytest.approx(100 - 30 - 0.5 - 0.485, abs=0.01)
    assert not r.hit_target and r.trades == 1


def test_gap_through_liquidation():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [90, 91, 89, 90], [90, 90, 90, 90]])
    r = run(bars, _once(Plan(1, 94.0, 10.0)), bars.index[0], days=1, capital=100, costs=C)
    assert r.liquidations == 1 and r.equity_end < 10


# --- Bot con cuenta simulada ---
import logging  # noqa: E402

import numpy as np  # noqa: E402

from phoenix.perps.bot import Bot, Config  # noqa: E402
from phoenix.perps.exchange import PaperPerp  # noqa: E402


def _trend_candles(n=1300, breakout_last=True):
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    close = 60000 + np.arange(n) * 20.0 + np.sin(np.arange(n) / 5) * 300
    if breakout_last:
        close[-1] = close[-60:-1].max() + 800  # cierra por encima del máximo de 20 velas
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 50
    low = np.minimum(open_, close) - 50
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def _bot(tmp_path, equity=114.0):
    x = PaperPerp(equity)
    return Bot(x, Config(), tmp_path / "state.json", logging.getLogger("test")), x


def test_bot_opens_long_with_stop_and_target(tmp_path):
    bot, x = _bot(tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    assert bot.step(c, now=c.index[-1] + pd.Timedelta(hours=4))
    assert x.pos is not None and x.pos.side == 1
    assert x.pos.qty * x.last <= 9.0 * 114.0 + 1e-6
    assert x.stop < x.last < x.tp
    # el objetivo está donde el capital llegaría a 5 x 114 $
    assert x.equity() + x.pos.qty * (x.tp - x.last) == pytest.approx(570, rel=0.01)


def test_bot_refuses_account_bigger_than_planned(tmp_path):
    bot, x = _bot(tmp_path, equity=1000.0)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    with pytest.raises(RuntimeError):
        bot.step(c)


def test_bot_stops_at_target_and_state_survives_restart(tmp_path):
    bot, x = _bot(tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    bot.step(c, now=c.index[-1] + pd.Timedelta(hours=4))
    x.cash = 600.0  # el capital supera el objetivo
    assert not bot.step(c, now=c.index[-1] + pd.Timedelta(hours=5))
    assert x.pos is None
    bot2 = Bot(x, Config(), tmp_path / "state.json", logging.getLogger("test"))
    assert bot2.st.finished and not bot2.step(c)
