import phoenix_config as config
from core.risk_manager import RiskManager


def test_reject_oversized_lot():
    rm = RiskManager(account_balance=200.0)
    ok, msg = rm.can_open_trade(
        symbol="XAUUSD",
        lot_size=config.MAX_LOT_SIZE + 0.01,
        stop_loss_pips=config.DEFAULT_STOP_LOSS_PIPS,
        take_profit_pips=30,
    )
    assert ok is False
    assert "LOTE" in msg


def test_reject_daily_loss_limit():
    rm = RiskManager(account_balance=200.0)
    rm.daily_pnl = -config.MAX_DAILY_LOSS_USD
    ok, msg = rm.can_open_trade(
        symbol="XAUUSD",
        lot_size=config.MIN_LOT_SIZE,
        stop_loss_pips=config.DEFAULT_STOP_LOSS_PIPS,
        take_profit_pips=30,
    )
    assert ok is False
    assert "LÍMITE DIARIO" in msg
