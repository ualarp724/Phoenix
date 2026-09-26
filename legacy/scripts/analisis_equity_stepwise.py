
import pandas as pd

def main():
	df = pd.read_csv('reports/equity_tridente_stepwise.csv', parse_dates=['timestamp'])
	equity_final = df['equity'].iloc[-1]
	equity_ini = df['equity'].iloc[0]
	profit = equity_final - equity_ini
	days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
	avg_daily = profit / days if days > 0 else 0
	max_dd = 0
	peak = df['equity'].iloc[0]
	for eq in df['equity']:
		if eq > peak:
			peak = eq
		dd = (peak - eq) / peak
		if dd > max_dd:
			max_dd = dd
	print(f"Equity inicial: ${equity_ini:.2f}")
	print(f"Equity final:   ${equity_final:.2f}")
	print(f"Profit total:   ${profit:.2f}")
	print(f"Días:           {days}")
	print(f"Profit diario:  ${avg_daily:.2f}")
	print(f"Max drawdown:   {max_dd*100:.2f}%")

if __name__ == '__main__':
	main()
