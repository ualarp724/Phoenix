import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"

INITIAL_EQUITY = 200.0
RISK_PCT = 0.01  # 1% riesgo compuesto
EQUITY_PROTECTOR = 0.15  # 15% caída máxima

# --- STOP LOSS PIPS POR ACTIVO (ajusta si tienes valores exactos) ---
SL_PIPS = {
    "BTCUSD": 200,    # ejemplo: 200 pips
    "NAS100": 100,    # ejemplo: 100 pips
    "XAUUSD": 20      # ejemplo: 20 pips
}
# --- VALOR DEL PIP POR MICRO LOTE (0.01) ---
PIP_VALUE = {
    "BTCUSD": 0.10,
    "NAS100": 0.10,
    "XAUUSD": 0.10
}

# --- CARGA Y UNIFICACIÓN DE TRADES ---
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

# --- SIMULACIÓN DE INTERÉS COMPUESTO ---
equity = INITIAL_EQUITY
equity_curve = []
max_equity = INITIAL_EQUITY
for i, row in trades.iterrows():
    symbol = row["symbol"]
    sl_pips = SL_PIPS[symbol]
    pip_value = PIP_VALUE[symbol]
    # Calcular lotaje dinámico
    risk_usd = equity * RISK_PCT
    lot = risk_usd / (sl_pips * pip_value)
    lot = max(0.01, round(lot, 2))
    # Escalar el pnl al nuevo lotaje
    pnl = row["pnl"] * (lot / 0.01)
    equity += pnl
    equity_curve.append((row["timestamp"], equity))
    max_equity = max(max_equity, equity)
    # Equity protector
    if equity < max_equity * (1 - EQUITY_PROTECTOR):
        print(f"Equity protector activado en {row['timestamp']}: equity={equity:.2f}")
        break

df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["equity"], label="Equity compuesta (1% interés compuesto)")
plt.title("Crecimiento Exponencial Tridente (BTC+NAS100+XAUUSD) - 1% riesgo dinámico")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/equity_tridente_compuesto.png", dpi=150)
plt.show()
print(f"Equity final: ${df['equity'].iloc[-1]:.2f}")
