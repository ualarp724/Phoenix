import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"

INITIAL_EQUITY = 200.0
EQUITY_PROTECTOR = 0.15  # 15% caída máxima (solo warning, no break)

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

# --- PARÁMETROS DE CRECIMIENTO GRADUAL ---
RISK_PCT_START = 0.006  # 0.6% riesgo inicial
RISK_PCT_MAX = 0.012    # 1.2% riesgo máximo
RISK_PCT_STEP = 0.0001  # Incremento de 0.01% cada $50 de equity
EQUITY_STEP = 50        # Cada $50 de equity, subir riesgo

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

# --- SIMULACIÓN DE CRECIMIENTO GRADUAL ---
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
    # Calcular lotaje gradual
    risk_usd = equity * risk_pct
    lot = risk_usd / (sl_pips * pip_value)
    lot = max(0.01, round(lot, 3))
    pnl = row["pnl"] * (lot / 0.01)
    equity += pnl
    equity_curve.append((row["timestamp"], equity, lot, risk_pct))
    max_equity = max(max_equity, equity)
    if equity < max_equity * (1 - EQUITY_PROTECTOR):
        print(f"Equity protector activado en {row['timestamp']}: equity={equity:.2f}")
        # No break: solo warning

df = pd.DataFrame(equity_curve, columns=["timestamp", "equity", "lot", "risk_pct"])
df.to_csv("reports/equity_tridente_gradual.csv", index=False)
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["equity"], label="Equity gradual", color="#27AE60")
plt.title("Crecimiento Tridente Gradual (BTC+NAS100+XAUUSD) - Lote gradual")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/equity_tridente_gradual.png", dpi=150)
plt.show()
print(f"Equity final: ${df['equity'].iloc[-1]:.2f}")
