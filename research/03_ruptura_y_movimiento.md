# Fase 3 — Segunda tanda: rupturas y modelo de movimiento

Generado por `research/03_ruptura_y_movimiento.py` el 26/09/2026. Mismas 9 ventanas y reglas que la primera tanda.

| Estrategia | Operaciones | Por mes | Acierto | R medio (IC 95 %) | PF | Ventanas en positivo | DD 95 % MC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Londres, TP 2x, 32 velas | 489 | 18 | 31.3 % | -0.106 (-0.231 a +0.030) | 0.86 | 33 % | 134 % |
| B2 Londres + tendencia H4 | 291 | 11 | 38.1 % | -0.089 (-0.228 a +0.057) | 0.87 | 33 % | 79 % |
| B3 Rango de apertura de Nueva York (9:30-10:30) | 293 | 11 | 41.0 % | -0.096 (-0.217 a +0.028) | 0.85 | 33 % | 73 % |
| B4 Donchian + filtro de movimiento (p70) | 961 | 36 | 37.7 % | -0.094 (-0.167 a -0.017) | 0.85 | 22 % | 204 % |
| B5 Londres + filtro de movimiento (p50) | 404 | 15 | 41.8 % | +0.007 (-0.116 a +0.126) | 1.03 | 44 % | 58 % |

## R medio por ventana

| Estrategia | 05/23 | 08/23 | 11/23 | 02/24 | 05/24 | 08/24 | 11/24 | 02/25 | 05/25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 Londres, TP 2x, 32 velas | -0.45 | -0.12 | -0.40 | +0.22 | +0.16 | -0.26 | -0.07 | +0.21 | -0.12 |
| B2 Londres + tendencia H4 | -0.29 | +0.04 | -0.02 | +0.21 | -0.07 | -0.20 | -0.10 | +0.23 | -0.56 |
| B3 Rango de apertura de Nueva York (9:30-10:30) | -0.11 | +0.02 | +0.01 | -0.12 | -0.20 | -0.04 | -0.36 | -0.06 | +0.25 |
| B4 Donchian + filtro de movimiento (p70) | -0.10 | -0.05 | +0.05 | -0.04 | -0.16 | -0.18 | -0.12 | +0.16 | -0.28 |
| B5 Londres + filtro de movimiento (p50) | -0.14 | +0.00 | -0.01 | +0.11 | +0.17 | -0.14 | -0.04 | +0.37 | -0.19 |