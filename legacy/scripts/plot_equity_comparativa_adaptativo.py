import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar equity original y adaptativo ---
df_orig = pd.read_csv("reports/equity_conjunta_3bots.csv", parse_dates=["timestamp"])
df_adapt = pd.read_csv("reports/equity_tridente_adaptativo.csv", parse_dates=["timestamp"])

plt.figure(figsize=(14,6))
plt.plot(df_orig["timestamp"], df_orig["equity"], label="Lote fijo (original)", color="#2E86C1")
plt.plot(df_adapt["timestamp"], df_adapt["equity"], label="Lote adaptativo", color="#E67E22")
plt.title("Comparativa Equity: Lote Fijo vs Adaptativo (BTC+NAS100+XAUUSD)")
plt.xlabel("Fecha")
plt.ylabel("Equity (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/equity_comparativa_adaptativo.png", dpi=150)
plt.show()
