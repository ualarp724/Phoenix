# Bot de alto riesgo: de 100 € a 500 € en 30 días (Test intocable)

Generado por `research/08_bot_arriesgado.py` el 27/09/2026. 269 ventanas de 30 días (inicios del 01/12/2025 al 26/08/2026); las ventanas se solapan, así que equivalen a unos 9 meses independientes. Perpetuo de BTC con 9x como máximo, comisión de taker 0,05 %, funding, liquidación.

| Estrategia | Llega a 500 € | Acaba perdiendo | Pierde ≥ 90 % | Resultado mediano | Resultado medio | Operaciones/mes |
| --- | --- | --- | --- | --- | --- | --- |
| S1 Ruptura 20/10, stop 2 ATR, riesgo 30 % | 0.0 % | 68 % | 0 % | 56 € | 85 € | 6.7 |
| S2 S1 + filtro de tendencia | 0.0 % | 68 % | 0 % | 71 € | 94 € | 4.5 |
| S3 S1 + piramidar | 20.4 % | 62 % | 1 % | 62 € | 166 € | 6.2 |
| S4 S1 + filtro + piramidar | 16.7 % | 65 % | 0 % | 69 € | 160 € | 4.3 |
| S5 Todo o nada a favor de la tendencia (9x, stop 4 %) | 0.0 % | 57 % | 0 % | 77 € | 104 € | 2.5 |
| S6 Control al azar (media de 5 semillas) | 0.1 % | 64 % | 0 % | 82 € | 99 € | 6.9 |

## Probabilidad de llegar a 500 € por año de inicio

| Estrategia | 2025 | 2026 |
| --- | --- | --- |
| S1 | 0 % | 0 % |
| S2 | 0 % | 0 % |
| S3 | 0 % | 23 % |
| S4 | 0 % | 19 % |
| S5 | 0 % | 0 % |
| S6 | 0 % | 0 % |

Tiempo de cálculo: 4 s.