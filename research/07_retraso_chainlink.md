# Retraso de Chainlink: ¿se puede aprovechar?

Generado por `research/07_retraso_chainlink.py` el 26/09/2026. 53,670 mercados (15/02 – 28/08/2026). Estrategia: apostar al lado que indica Coinbase frente a price_to_beat cuando la diferencia es grande. Rentabilidad por dólar apostado, con comisión, al precio medio pagado por los takers en cada ventana.

| Diferencia | Mercados | Acierto | +0..+2 s | +1..+5 s | +5..+15 s |
| --- | --- | --- | --- | --- | --- |
| ≥ 9 $ (el 20 % mayor) | 10,717 | 54.3 % | +0.6 % (± 1.9) | -1.1 % (± 1.8) | -2.7 % (± 1.8) |
| ≥ 12 $ (el 10 % mayor) | 5,359 | 56.4 % | +2.2 % (± 2.6) | +0.3 % (± 2.5) | -1.6 % (± 2.5) |
| ≥ 17 $ (el 5 % mayor) | 2,680 | 58.8 % | +3.6 % (± 3.5) | +1.0 % (± 3.4) | -1.2 % (± 3.4) |

La ventaja solo aparece en los dos primeros segundos y se evapora en cuanto pasan unos pocos: es una carrera de latencia contra bots que ven el feed de Chainlink en tiempo real. price_to_beat no se publica durante la ventana.