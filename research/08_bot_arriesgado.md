# Bot de alto riesgo: de 100 € a 500 € en 30 días (Desarrollo)

Generado por `research/08_bot_arriesgado.py` el 27/09/2026. 2345 ventanas de 30 días (inicios del 01/07/2019 al 30/11/2025); las ventanas se solapan, así que equivalen a unos 78 meses independientes. Perpetuo de BTC con 9x como máximo, comisión de taker 0,05 %, funding, liquidación.

| Estrategia | Llega a 500 € | Acaba perdiendo | Pierde ≥ 90 % | Resultado mediano | Resultado medio | Operaciones/mes |
| --- | --- | --- | --- | --- | --- | --- |
| S1 Ruptura 20/10, stop 2 ATR, riesgo 30 % | 2.5 % | 61 % | 0 % | 82 € | 108 € | 6.0 |
| S2 S1 + filtro de tendencia | 3.3 % | 58 % | 0 % | 91 € | 121 € | 4.3 |
| S3 S1 + piramidar | 9.9 % | 67 % | 3 % | 68 € | 119 € | 6.0 |
| S4 S1 + filtro + piramidar | 11.0 % | 63 % | 2 % | 81 € | 131 € | 4.4 |
| S5 Todo o nada a favor de la tendencia (9x, stop 4 %) | 3.6 % | 60 % | 8 % | 74 € | 121 € | 2.9 |
| S6 Control al azar (media de 5 semillas) | 1.2 % | 62 % | 0 % | 80 € | 106 € | 6.6 |

## Probabilidad de llegar a 500 € por año de inicio

| Estrategia | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 0 % | 12 % | 0 % | 0 % | 0 % | 4 % | 0 % |
| S2 | 0 % | 12 % | 0 % | 3 % | 1 % | 5 % | 0 % |
| S3 | 4 % | 17 % | 5 % | 13 % | 9 % | 13 % | 4 % |
| S4 | 0 % | 19 % | 5 % | 18 % | 10 % | 15 % | 4 % |
| S5 | 0 % | 13 % | 4 % | 0 % | 0 % | 7 % | 0 % |
| S6 | 1 % | 4 % | 0 % | 1 % | 1 % | 1 % | 0 % |

Tiempo de cálculo: 35 s.