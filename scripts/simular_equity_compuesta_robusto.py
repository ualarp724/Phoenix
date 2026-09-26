import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"

INITIAL_EQUITY = 200.0
EQUITY_PROTECTOR = 0.35  # 35% caída máxima (stop duro)
MIN_EQUITY = 120.0       # No dejar bajar de $120 (stop total)

# --- STOP LOSS PIPS POR ACTIVO ---
SL_PIPS = {
    "BTCUSD": 200,
    "NAS100": 100,
    "XAUUSD": 20
}
# --- VALOR DEL PIP POR MICRO LOTE (0.01) ---
PIP_VALUE = {
    "BTCUSD": 0.10,
    "NAS100": 0.10,
    "XAUUSD": 0.10
}

# --- PARÁMETROS DE LOTAJE ROBUSTO ---
RISK_PCT_START = 0.012   # 1.2% riesgo inicial
RISK_PCT_MAX = 0.022     # 2.2% riesgo máximo
RISK_PCT_STEP = 0.00025  # Incremento de 0.025% cada $30 equity
EQUITY_STEP = 30         # Cada $30 equity, subir riesgo
LOT_MAX = 0.022          # Límite absoluto de lotaje

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

# --- SIMULACIÓN DE LOTAJE ROBUSTO ---
equity = INITIAL_EQUITY
max_equity = INITIAL_EQUITY
risk_pct = RISK_PCT_START
equity_curve = []
for i, row in trades.iterrows():
    symbol = row["symbol"]
    sl_pips = SL_PIPS[symbol]
    pip_value = PIP_VALUE[symbol]
    # Ajustar riesgo cada EQUITY_STEP
    if equity > INITIAL_EQUITY:
        steps = int((equity - INITIAL_EQUITY) // EQUITY_STEP)
        risk_pct = min(RISK_PCT_START + steps * RISK_PCT_STEP, RISK_PCT_MAX)
    # Calcular lotaje robusto
    risk_usd = equity * risk_pct
    lot = risk_usd / (sl_pips * pip_value)
    lot = max(0.01, min(round(lot, 3), LOT_MAX))
    pnl = row["pnl"] * (lot / 0.01)
    equity += pnl
    equity_curve.append((row["timestamp"], equity, lot, risk_pct))
    max_equity = max(max_equity, equity)
    # Stop duro por drawdown
    if equity < max_equity * (1 - EQUITY_PROTECTOR) or equity < MIN_EQUITY:
        print(f"STOP: Equity protector o mínimo alcanzado en {row['timestamp']}: equity={equity:.2f}")
        break

df = pd.DataFrame(equity_curve, columns=["timestamp", "equity", "lot", "risk_pct"])
df.to_csv("reports/equity_tridente_robusto.csv", index=False)
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["equity"], label="Equity robusto", color="#884EA0")
plt.title("Crecimiento Tridente Robusto (BTC+NAS100+XAUUSD) - Lote robusto")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/equity_tridente_robusto.png", dpi=150)
plt.show()
print(f"Equity final: ${df['equity'].iloc[-1]:.2f}")
