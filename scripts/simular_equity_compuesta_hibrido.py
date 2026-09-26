import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"

INITIAL_EQUITY = 200.0
SAFE_EQUITY = 300.0  # Hasta aquí solo lote fijo
SAFE_LOT = 0.01      # Lote fijo ultra conservador
EQUITY_PROTECTOR = 0.60  # 60% caída máxima (stop duro)
MIN_EQUITY = 80.0        # No dejar bajar de $80 (stop total)

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

# --- PARÁMETROS DE LOTAJE PROGRESIVO ---
RISK_PCT_START = 0.008   # 0.8% riesgo inicial
RISK_PCT_MAX = 0.025     # 2.5% riesgo máximo
RISK_PCT_STEP = 0.00015  # Incremento de 0.015% cada $25 equity
EQUITY_STEP = 25         # Cada $25 equity, subir riesgo
LOT_MAX = 0.025          # Límite absoluto de lotaje

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

# --- SIMULACIÓN HÍBRIDA: LOTE FIJO + PROGRESIVO ---
equity = INITIAL_EQUITY
max_equity = INITIAL_EQUITY
risk_pct = RISK_PCT_START
equity_curve = []
for i, row in trades.iterrows():
    symbol = row["symbol"]
    sl_pips = SL_PIPS[symbol]
    pip_value = PIP_VALUE[symbol]
    # 1. Lote fijo ultra seguro hasta SAFE_EQUITY
    if equity < SAFE_EQUITY:
        lot = SAFE_LOT
    else:
        # 2. Progresivo a partir de $300
        steps = int((equity - SAFE_EQUITY) // EQUITY_STEP)
        risk_pct = min(RISK_PCT_START + steps * RISK_PCT_STEP, RISK_PCT_MAX)
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
df.to_csv("reports/equity_tridente_hibrido.csv", index=False)
plt.figure(figsize=(14,6))
plt.plot(df["timestamp"], df["equity"], label="Equity híbrido (fijo+progresivo)", color="#1F618D")
plt.title("Crecimiento Tridente Híbrido (BTC+NAS100+XAUUSD) - Lote fijo y progresivo")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/equity_tridente_hibrido.png", dpi=150)
plt.show()
print(f"Equity final: ${df['equity'].iloc[-1]:.2f}")
