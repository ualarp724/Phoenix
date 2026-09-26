"""Bot para los mercados "Bitcoin Up or Down - 5 minutos" de Polymarket.

Módulos:
- binance: velas de 1 minuto de BTCUSDT (fuente de las features).
- polymarket: lista de mercados, resultado real y operaciones alrededor del inicio.
- features: features en el momento de decidir, solo con velas ya cerradas.
- backtest: apuestas simuladas a precios reales de Polymarket, con comisiones.
"""
