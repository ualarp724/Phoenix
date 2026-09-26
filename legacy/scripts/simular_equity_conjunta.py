import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Paths a los CSV exportados
BTC_CSV = "reports/trades_btc_2024_2025.csv"
NAS100_CSV = "reports/trades_nas100_2024_2025.csv"
XAUUSD_CSV = "reports/oro_2025-07_to_2026-01_trades.csv"  # Ajusta si tu CSV de oro tiene otro nombre

# Equity inicial
INITIAL_EQUITY = 200.0

# Leer trades
btc = pd.read_csv(BTC_CSV, parse_dates=["timestamp"])
btc["bot"] = "BTC"
nas = pd.read_csv(NAS100_CSV, parse_dates=["timestamp"])
nas["bot"] = "NAS100"
xau = pd.read_csv(XAUUSD_CSV, parse_dates=["timestamp"])

all_trades = pd.concat([btc, nas, xau], ignore_index=True)
all_trades = all_trades.sort_values("timestamp").reset_index(drop=True)

# Simular equity conjunta
all_trades["equity"] = INITIAL_EQUITY + all_trades["pnl"].cumsum()

# Graficar equity y drawdown
plt.figure(figsize=(14,6))
plt.plot(all_trades["timestamp"], all_trades["equity"], label="Equity conjunta", color="#2E86C1")

# Calcular drawdown
peak = all_trades["equity"].cummax()
drawdown = (all_trades["equity"] - peak)
plt.fill_between(all_trades["timestamp"], all_trades["equity"], peak, color="red", alpha=0.15, label="Drawdown")

plt.title("Evolución Equity Combinada (BTC + NAS100 + XAUUSD)")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/equity_conjunta_3bots.png", dpi=150)
plt.show()

# Guardar equity/drawdown a CSV
all_trades[["timestamp", "equity"]].to_csv("reports/equity_conjunta_3bots.csv", index=False)
print("Equity y drawdown combinados exportados a reports/equity_conjunta_3bots.csv y .png")
