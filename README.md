# Phoenix — bot de trading para XAUUSD

Proyecto en reconstrucción siguiendo el plan "Plan Phoenix: bot de oro".
Objetivo: comprobar con pruebas honestas si hay una estrategia en oro (M15)
que gane dinero después de costes, y solo entonces operarla.

## Estructura

| Carpeta | Qué contiene |
| --- | --- |
| `phoenix/` | Código de verdad: datos, costes, features, etiquetas, backtest, modelo |
| `research/` | Experimentos e informes de investigación (fase 3) |
| `live/` | Ejecución en vivo: runner de Python y EA puente de MT5 (fase 5) |
| `tests/` | Tests con pytest |
| `config/` | Configuración (activo, costes, riesgo) |
| `data/raw/` | Datos exportados de MT5. No van a git |
| `legacy/` | Código, modelos, docs y resultados antiguos. Solo referencia |

`legacy/` no se mantiene: sus resultados tienen fugas de datos del futuro
(ver la auditoría del proyecto). Las piezas útiles se migran a `phoenix/`
una a una, con tests. Sus rutas a los CSV ya no funcionan porque los datos
están ahora en `data/raw/`.

## Entorno

Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Datos en `data/raw/`

| Archivo | Activo | Marco | Uso |
| --- | --- | --- | --- |
| `vantage_gold.csv` | XAUUSD | M15 | Principal |
| `m5.csv` | XAUUSD | M5 | Apoyo |
| `h1.csv` | XAUUSD | H1 | Apoyo (desde 2018) |
| `vantage_live_gold.csv` | XAUUSD | — | Instantánea antigua del live |
| `vantage_btc1.csv`, `vantage_btc2.csv` | BTCUSD | M5 / M15 | Otros activos, más adelante |
| `vantage_nas100.csv` | NAS100 | M15 | Otros activos, más adelante |
| `vantage_eurusd.csv` | EURUSD | M15 | Otros activos, más adelante |

Todas las horas se pasan a UTC (el servidor de Vantage va con Nueva York + 7 h).

Periodos (ver `config/xauusd.json`):

- **Desarrollo:** 15/11/2021 – 31/07/2025. Aquí se investiga (fase 3).
- **Test intocable:** 01/08/2025 – 06/02/2026. Solo se abre en la fase 4 con
  `holdout_data(..., i_know_this_is_the_final_test=True)`.
- **Test 2:** feb–sep 2026, cuando se exporte de MT5.

Informe de datos y costes: `python research/01_datos.py` → `research/01_datos.md`.
