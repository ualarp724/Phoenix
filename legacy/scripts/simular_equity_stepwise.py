import pandas as pd
import matplotlib.pyplot as plt

BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"

INITIAL_EQUITY = 200.0
EQUITY_PROTECTOR = 0.15

SL_PIPS = {
    "BTCUSD": 200,
    "NAS100": 100,
    "XAUUSD": 20
}
PIP_VALUE = {
    "BTCUSD": 0.10,
    "NAS100": 0.10,
    "XAUUSD": 0.10
}
RISK_PCT = 0.01
STEP_TRIGGER = 100  # Subir lote cada $100 de equity extra
STEP_INCREASE = 0.10  # +10% lote por escalón

btc = pd.read_csv(BTC_CSV, parse_dates=["timestamp"])
btc["symbol"] = "BTCUSD"
nas = pd.read_csv(NAS100_CSV, parse_dates=["timestamp"])
nas["symbol"] = "NAS100"
xau = pd.read_csv(XAUUSD_CSV, parse_dates=["timestamp"])
xau["symbol"] = "XAUUSD"
if "pnl" not in xau.columns and "profit" in xau.columns:
    xau = xau.rename(columns={"profit": "pnl"})
trades = pd.concat([
    btc[["timestamp", "pnl", "symbol"]],
    nas[["timestamp", "pnl", "symbol"]],
    xau[["timestamp", "pnl", "symbol"]],
], ignore_index=True)
trades = trades.sort_values("timestamp").reset_index(drop=True)

# Calcular lote base (fijo hasta $300)
def calc_lote_base(symbol):
    sl_pips = SL_PIPS[symbol]
    pip_value = PIP_VALUE[symbol]
    risk_usd = INITIAL_EQUITY * RISK_PCT
    lot = risk_usd / (sl_pips * pip_value)
    return max(0.01, round(lot, 2))

lote_base = {s: calc_lote_base(s) for s in SL_PIPS}
lote_actual = lote_base.copy()
step_level = 0
max_equity = INITIAL_EQUITY
equity = INITIAL_EQUITY
equity_curve = []

for i, row in trades.iterrows():
    symbol = row["symbol"]
    # Escalado: solo subir lote si equity supera múltiplos de $100
    if equity >= 300:
        new_level = int((equity - 200) // STEP_TRIGGER)
        if new_level > step_level:
            # Subir lote un 10% por cada $100 extra
            lote_actual = {s: lote_actual[s] * (1 + STEP_INCREASE) for s in lote_actual}
            step_level = new_level
    # Usar el lote actual (no baja si equity baja)
    lot = lote_actual[symbol]
    pnl = row["pnl"] * (lot / 0.01)
    equity += pnl
    equity_curve.append((row["timestamp"], equity, lot))
    max_equity = max(max_equity, equity)
    if equity < max_equity * (1 - EQUITY_PROTECTOR):
        print(f"Equity protector activado en {row['timestamp']}: equity={equity:.2f}")
        # No break: dejar que la simulación continúe aunque la cuenta caiga

df = pd.DataFrame(equity_curve, columns=["timestamp", "equity", "lot"])
df.to_csv("reports/equity_tridente_stepwise.csv", index=False)
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["equity"], label="Equity stepwise")
plt.title("Crecimiento Tridente Stepwise (BTC+NAS100+XAUUSD) - Lote escalonado")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/equity_tridente_stepwise.png", dpi=150)
plt.show()
print(f"Equity final: ${df['equity'].iloc[-1]:.2f}")
