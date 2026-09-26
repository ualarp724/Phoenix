# ¿Se puede predecir BTC a 5 minutos? (idea Polymarket)

Generado por `research/04_btc_5min.py` el 26/09/2026. Velas de 1 min de Binance, 10 ventanas de validación de 3 meses (01/2024 → 05/2026). Para ganar dinero hay que acertar más del **52.25 %** (entrada a 0.505 + comisión de 0.0175), suponiendo que se puede comprar a 0,505 justo al empezar.


## Objetivo «cierre»

| Estrategia | Apuestas | Por día | Acierto (IC 95 %) | Ganancia media por acción |
| --- | --- | --- | --- | --- |
| Siempre «sube» | 253,727 | 288 | 50.25 % (± 0.19) | -2.00 c |
| Momentum (repetir los últimos 5 min) | 253,727 | 288 | 49.56 % (± 0.19) | -2.69 c |
| LightGBM, todas | 253,727 | 288 | 51.64 % (± 0.19) | -0.61 c |
| LightGBM, solo el 20 % más seguro | 54,393 | 62 | 54.17 % (± 0.42) | +1.92 c |
| LightGBM, solo el 10 % más seguro | 27,872 | 32 | 55.06 % (± 0.58) | +2.81 c |
| LightGBM, solo el 5 % más seguro | 14,243 | 16 | 55.53 % (± 0.82) | +3.28 c |
| LightGBM, solo el 1 % más seguro | 2,976 | 3 | 56.85 % (± 1.78) | +4.61 c |

Acierto del 5 % más seguro por ventana: 55.4 %, 56.4 %, 55.3 %, 55.1 %, 54.8 %, 54.9 %, 54.4 %, 56.6 %, 57.5 %, 56.0 %. Features con más peso: `pos60` 9 %, `r60` 8 %, `vol60` 7 %, `r15` 7 %, `r5` 7 %.

## Objetivo «twap_ventana»

| Estrategia | Apuestas | Por día | Acierto (IC 95 %) | Ganancia media por acción |
| --- | --- | --- | --- | --- |
| Siempre «sube» | 253,727 | 288 | 49.96 % (± 0.19) | -2.29 c |
| Momentum (repetir los últimos 5 min) | 253,727 | 288 | 52.02 % (± 0.19) | -0.23 c |
| LightGBM, todas | 253,727 | 288 | 56.23 % (± 0.19) | +3.98 c |
| LightGBM, solo el 20 % más seguro | 52,760 | 60 | 63.85 % (± 0.41) | +11.60 c |
| LightGBM, solo el 10 % más seguro | 26,558 | 30 | 66.61 % (± 0.57) | +14.36 c |
| LightGBM, solo el 5 % más seguro | 13,160 | 15 | 68.55 % (± 0.79) | +16.30 c |
| LightGBM, solo el 1 % más seguro | 2,636 | 3 | 71.70 % (± 1.72) | +19.45 c |

Acierto del 5 % más seguro por ventana: 64.9 %, 68.1 %, 71.4 %, 69.8 %, 68.9 %, 71.2 %, 69.9 %, 67.9 %, 65.1 %, 67.2 %. Features con más peso: `r1` 44 %, `upwick5` 6 %, `body5` 6 %, `pos60` 5 %, `r15` 5 %.

## Objetivo «twap_final»

| Estrategia | Apuestas | Por día | Acierto (IC 95 %) | Ganancia media por acción |
| --- | --- | --- | --- | --- |
| Siempre «sube» | 253,727 | 288 | 50.23 % (± 0.19) | -2.02 c |
| Momentum (repetir los últimos 5 min) | 253,727 | 288 | 51.28 % (± 0.19) | -0.97 c |
| LightGBM, todas | 253,727 | 288 | 54.22 % (± 0.19) | +1.97 c |
| LightGBM, solo el 20 % más seguro | 53,472 | 61 | 59.32 % (± 0.42) | +7.07 c |
| LightGBM, solo el 10 % más seguro | 27,129 | 31 | 60.89 % (± 0.58) | +8.64 c |
| LightGBM, solo el 5 % más seguro | 13,741 | 16 | 62.23 % (± 0.81) | +9.98 c |
| LightGBM, solo el 1 % más seguro | 2,771 | 3 | 64.27 % (± 1.78) | +12.02 c |

Acierto del 5 % más seguro por ventana: 59.2 %, 61.6 %, 64.8 %, 62.6 %, 62.3 %, 62.5 %, 62.1 %, 62.2 %, 62.6 %, 63.1 %. Features con más peso: `r1` 28 %, `upwick5` 7 %, `r60` 6 %, `r15` 6 %, `vol60` 6 %.