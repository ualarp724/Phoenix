from dataclasses import dataclass

import phoenix_config as config


@dataclass
class ExecutionResult:
    fill_price: float
    slippage_usd: float
    spread_usd: float


def apply_execution_costs(
    price: float,
    lot_size: float,
    side: str,
    spread_pips: float = 2.0,
    slippage_pips: float = 1.0,
    pip_value_usd: float = 0.10,
) -> ExecutionResult:
    """
    Simula spread y slippage básicos. side: 'BUY' | 'SELL'.
    """
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    spread_cost = spread_pips * pip_value_usd * (lot_size / config.MIN_LOT_SIZE)
    slippage_cost = slippage_pips * pip_value_usd * (lot_size / config.MIN_LOT_SIZE)

    if side == "BUY":
        fill_price = price + (spread_pips + slippage_pips) * 0.0001
    else:
        fill_price = price - (spread_pips + slippage_pips) * 0.0001

    return ExecutionResult(
        fill_price=fill_price,
        slippage_usd=slippage_cost,
        spread_usd=spread_cost,
    )
