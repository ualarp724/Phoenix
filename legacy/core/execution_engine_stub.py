from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.risk_manager import RiskManager
from core.logging_utils import log_event


@dataclass
class OrderRequest:
    symbol: str
    side: str
    lot_size: float
    stop_loss_pips: int
    take_profit_pips: int
    timestamp: datetime | None = None


class ExecutionEngine:
    """Stub de ejecución: valida riesgos y registra eventos."""

    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    def submit_order(self, request: OrderRequest) -> Optional[str]:
        ok, msg = self.risk_manager.can_open_trade(
            request.symbol,
            request.lot_size,
            request.stop_loss_pips,
            request.take_profit_pips,
            current_time=request.timestamp,
        )
        if not ok:
            log_event({"event": "ORDER_REJECTED", "reason": msg, "symbol": request.symbol})
            return None

        self.risk_manager.record_trade_opened(request.symbol, request.lot_size)
        log_event({"event": "ORDER_ACCEPTED", "symbol": request.symbol, "side": request.side})
        return "SIMULATED_ORDER_ID"
