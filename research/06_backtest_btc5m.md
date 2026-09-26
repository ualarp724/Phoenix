# Bot BTC 5m — backtest con precios reales de Polymarket (Desarrollo)

Generado por `research/06_backtest_btc5m.py` el 26/09/2026. 56,147 mercados resueltos (15/02/2026 → 28/08/2026). Apuesta fija de 5 $, comisión 0,07·p·(1−p). El objetivo de Binance coincide con el resultado real en el 91.1 % de estos mercados.

| Configuración | Apuestas | Por día | Acierto | Precio medio | Ventaja esperada | Ganancia (5 $/apuesta) | Rentabilidad por apuesta | Días en positivo | Peor caída |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 inicio, +1..+5 s, margen 0,02 | 29,410 | 152 | 46.0 % | 0.476 | +0.065 | -10,890 $ | -7.4 % | 17 % | 10,822 $ |
| C2 inicio, +1..+10 s, margen 0,02 | 32,113 | 166 | 43.7 % | 0.459 | +0.073 | -14,717 $ | -9.2 % | 16 % | 14,625 $ |
| C3 inicio, +1..+5 s, margen 0,05 | 15,699 | 81 | 44.0 % | 0.461 | +0.092 | -7,039 $ | -9.0 % | 21 % | 7,056 $ |
| C4 1 min antes, -55..-5 s, margen 0,02 | 8,459 | 44 | 44.9 % | 0.459 | +0.053 | -2,894 $ | -6.8 % | 38 % | 2,910 $ |

## Calibración (decisión en el inicio, precio de +1..+5 s)

¿Acierta el modelo lo que dice? ¿Y el mercado ya lo sabía? Si el precio de «Up» sigue a la frecuencia real, el mercado ya lo descuenta.

| Prob. del modelo | Mercados | Media modelo | Frecuencia real de «Up» | Precio medio de «Up» |
| --- | --- | --- | --- | --- |
| (0.2, 0.35] | 288 | 0.330 | 0.514 | 0.492 |
| (0.35, 0.45] | 12,248 | 0.421 | 0.496 | 0.496 |
| (0.45, 0.55] | 29,986 | 0.498 | 0.497 | 0.508 |
| (0.55, 0.65] | 12,161 | 0.582 | 0.513 | 0.521 |
| (0.65, 0.8] | 452 | 0.672 | 0.566 | 0.534 |

## Resultado por mes (C1)

| Mes | Apuestas | Acierto | Ganancia |
| --- | --- | --- | --- |
| 2026-02 | 2031 | 46.7 % | -876 $ |
| 2026-03 | 4543 | 47.2 % | -1615 $ |
| 2026-04 | 4528 | 45.6 % | -2033 $ |
| 2026-05 | 4749 | 46.4 % | -1486 $ |
| 2026-06 | 4811 | 46.1 % | -1949 $ |
| 2026-07 | 4897 | 47.3 % | -1537 $ |
| 2026-08 | 3851 | 42.5 % | -1394 $ |

Tiempo de cálculo: 111 s.