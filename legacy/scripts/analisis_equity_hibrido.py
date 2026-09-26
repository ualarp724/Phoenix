import pandas as pd

# Leer equity híbrido
# (El archivo tiene columnas: timestamp, equity, lot, risk_pct)
df = pd.read_csv("reports/equity_tridente_hibrido.csv", parse_dates=["timestamp"])

# Calcular profit diario
# Tomar el último equity de cada día
df["date"] = df["timestamp"].dt.date
daily_equity = df.groupby("date")["equity"].last()
daily_profit = daily_equity.diff().dropna()

avg_daily_profit = daily_profit.mean()
total_days = len(daily_profit)

# Calcular drawdown máximo
peak = df["equity"].cummax()
drawdown = df["equity"] - peak
max_drawdown = drawdown.min()
max_drawdown_pct = (max_drawdown / peak.max()) * 100

print(f"Promedio diario: ${avg_daily_profit:.2f} USD ({total_days} días)")
print(f"Drawdown máximo: ${max_drawdown:.2f} USD ({max_drawdown_pct:.2f}%)")
