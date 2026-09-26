# Fase 3 — Primera tanda: walk-forward en desarrollo (XAUUSD M15)

Generado por `research/02_walkforward.py` el 26/09/2026. 9 ventanas de validación de 3 meses (15/05/2023 → 01/08/2025), cada una entrenada con los 18 meses anteriores. SL = 1 ATR (máx. 5.70 $/oz por el lote mínimo), riesgo fijo de 5.70 $, costes reales.

| Estrategia | Operaciones | Por mes | Acierto | R medio (IC 95 %) | PF | Ventanas en positivo | DD 95 % MC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Azar (semilla 0) | 1228 | 46 | 37.1 % | -0.137 (-0.200 a -0.073) | 0.81 | 11 % | 288 % |
| Azar (semilla 1) | 1256 | 47 | 34.2 % | -0.213 (-0.277 a -0.149) | 0.67 | 0 % | 505 % |
| Azar (semilla 2) | 1264 | 48 | 36.3 % | -0.181 (-0.243 a -0.116) | 0.71 | 0 % | 442 % |
| Azar (semilla 3) | 1214 | 46 | 37.3 % | -0.137 (-0.202 a -0.073) | 0.80 | 11 % | 294 % |
| Azar (semilla 4) | 1225 | 46 | 36.2 % | -0.157 (-0.222 a -0.094) | 0.77 | 11 % | 346 % |
| Ruptura Donchian 32 + EMA200 | 1973 | 74 | 38.1 % | -0.102 (-0.153 a -0.046) | 0.85 | 11 % | 377 % |
| Ruptura rango de Londres | 489 | 18 | 40.9 % | -0.021 (-0.125 a +0.090) | 0.98 | 33 % | 73 % |
| LightGBM SL 1 ATR, TP 1.5x, 16 velas, umbral 0.49 | 337 | 13 | 39.5 % | -0.063 (-0.189 a +0.072) | 0.97 | 44 % | 62 % |
| LightGBM SL 1 ATR, TP 2x, 32 velas, umbral 0.42 | 751 | 28 | 29.8 % | -0.149 (-0.247 a -0.050) | 0.82 | 33 % | 219 % |
| LightGBM SL 1 ATR, TP 1.5x, 16 velas, umbral 0.54 | 64 | 2 | 40.6 % | -0.045 (-0.351 a +0.269) | 0.97 | 67 % | 27 % |

## R medio por ventana

| Estrategia | 05/23 | 08/23 | 11/23 | 02/24 | 05/24 | 08/24 | 11/24 | 02/25 | 05/25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Azar (semilla 0) | -0.28 | -0.15 | -0.08 | -0.18 | +0.01 | -0.09 | -0.19 | -0.05 | -0.23 |
| Azar (semilla 1) | -0.33 | -0.32 | -0.25 | -0.28 | -0.18 | -0.09 | -0.13 | -0.02 | -0.18 |
| Azar (semilla 2) | -0.29 | -0.25 | -0.18 | -0.17 | -0.22 | -0.15 | -0.09 | -0.05 | -0.09 |
| Azar (semilla 3) | -0.24 | -0.18 | -0.22 | -0.08 | +0.07 | -0.12 | -0.15 | -0.28 | -0.04 |
| Azar (semilla 4) | -0.22 | -0.25 | -0.25 | -0.25 | -0.16 | -0.13 | -0.06 | -0.13 | +0.09 |
| Ruptura Donchian 32 + EMA200 | -0.20 | -0.16 | -0.00 | -0.12 | -0.11 | -0.12 | -0.05 | +0.05 | -0.15 |
| Ruptura rango de Londres | -0.21 | -0.11 | -0.04 | +0.14 | +0.17 | -0.16 | -0.01 | +0.32 | -0.21 |
| LightGBM SL 1 ATR, TP 1.5x, 16 velas, umbral 0.49 | -0.09 | -0.30 | -0.01 | +0.17 | +0.02 | -0.04 | -0.18 | +0.01 | +0.18 |
| LightGBM SL 1 ATR, TP 2x, 32 velas, umbral 0.42 | -0.29 | -0.49 | -0.25 | +0.07 | -0.33 | +0.19 | -0.14 | +0.02 | -0.11 |
| LightGBM SL 1 ATR, TP 1.5x, 16 velas, umbral 0.54 | +0.09 | -0.34 | +1.47 | -1.03 | +0.39 | +0.85 | -0.53 | +0.85 | +0.22 |

## Features más usadas por LightGBM (SL 1 ATR, TP 1,5x)

| Feature | Peso |
| --- | --- |
| `hour_sin` | 19.4 % |
| `hour_cos` | 14.1 % |
| `h4_dist_ema50` | 9.4 % |
| `h4_ret` | 8.8 % |
| `vol_regime` | 5.4 % |
| `h1_dist_ema50` | 4.9 % |
| `natr` | 4.9 % |
| `tick_vol_rel` | 4.6 % |

Tiempo de cálculo: 13 s.