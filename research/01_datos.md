# Fase 1 — Datos y costes de XAUUSD M15

Archivo `data/raw/vantage_gold.csv`, horas en UTC. Generado por `research/01_datos.py`.

## Integridad

| Comprobación | Resultado |
| --- | --- |
| Velas | 99,998 (2021-11-15 → 2026-02-06) |
| Duplicados / desordenadas / OHLC imposible / precios ≤ 0 | 0 / 0 / 0 / 0 |
| Velas a las 17:00 de Nueva York (pausa diaria; debe ser 0 si la conversión horaria es correcta) | 0 |
| Huecos no explicados por pausa diaria o fin de semana | 39 |
| Velas en desarrollo / test | 87,741 / 12,257 |

Los 10 huecos más largos (casi todos festivos):

| Reanuda (UTC) | Duración |
| --- | --- |
| 2025-04-20 22:00 | 3 days 01:15:00 |
| 2022-04-17 22:00 | 3 days 01:15:00 |
| 2023-04-09 22:00 | 3 days 01:15:00 |
| 2021-12-26 23:00 | 3 days 01:15:00 |
| 2024-03-31 22:00 | 3 days 01:15:00 |
| 2025-12-25 23:00 | 1 days 04:30:00 |
| 2024-12-25 23:00 | 1 days 04:30:00 |
| 2025-01-01 23:00 | 1 days 01:15:00 |
| 2026-01-01 23:00 | 1 days 01:15:00 |
| 2021-11-25 23:00 | 0 days 05:30:00 |

## Volatilidad y spread por año (solo desarrollo)

Riesgo máximo por operación: 2.5 % de 228 $ = 5.70 $ → SL máximo con 0,01 lotes = 5.70 $/oz.

| Año | Precio mediano ($) | ATR14 M15 mediano ($/oz) | Spread vela mediano (pts) | Spread usado ($/oz) | SL máx / ATR |
| --- | --- | --- | --- | --- | --- |
| 2021 | 1,793 | 1.63 | 2 | 0.12 | 3.50 |
| 2022 | 1,805 | 2.05 | 1 | 0.12 | 2.78 |
| 2023 | 1,945 | 1.79 | 18 | 0.18 | 3.18 |
| 2024 | 2,381 | 2.63 | 7 | 0.12 | 2.17 |
| 2025 | 3,220 | 4.18 | 7 | 0.12 | 1.36 |

## Coste de una operación con 0,01 lotes

Spread 0.12 $ + comisión 0.06 $ + deslizamiento 0.10 $ (2 ejecuciones a mercado) = **0.28 $**, un 4.9 % del riesgo máximo por operación.
