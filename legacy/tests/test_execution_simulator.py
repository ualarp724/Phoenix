from core.execution_simulator import apply_execution_costs


def test_execution_simulator_buy():
    result = apply_execution_costs(price=1.1000, lot_size=0.01, side="BUY")
    assert result.fill_price > 1.1000


def test_execution_simulator_sell():
    result = apply_execution_costs(price=1.1000, lot_size=0.01, side="SELL")
    assert result.fill_price < 1.1000
