#!/usr/bin/env python3
"""
Optuna Multi-Asset Optimizer: BTC, NAS100, XAUUSD
Optimiza los 3 modelos a la vez para maximizar profit, winrate y reducir drawdown.
Guarda logs estructurados de cada trial y los mejores parámetros.
"""
import os
import json
import logging
from datetime import datetime
import optuna
from typing import Dict, Any

# Configuración de logging estructurado
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/optuna_multiasset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("optuna_multiasset")

# Importar funciones de cada modelo

from optuna_nas100_vantage_m15 import objective as nas100_objective
from optuna_xauusd_vantage_m15 import objective as xauusd_objective
from optuna_btc_vantage2_m15 import objective as btc_objective

# Wrapper para optimizar NAS100 y XAUUSD a la vez
def multiasset_objective(trial: optuna.Trial) -> float:
    # NAS100 params
    params_nas100 = {
        'sl_mult': trial.suggest_float('nas100_sl_mult', 0.8, 3.0),
        'tp_rr': trial.suggest_float('nas100_tp_rr', 2.0, 3.0),
        'umbral_buy': trial.suggest_float('nas100_umbral_buy', 0.10, 0.30),
        'umbral_sell': trial.suggest_float('nas100_umbral_sell', 0.10, 0.30),
    }
    # XAUUSD params
    params_xauusd = {
        'sl_mult': trial.suggest_float('xauusd_sl_mult', 1.0, 3.5),
        'tp_rr': trial.suggest_float('xauusd_tp_rr', 2.3, 3.0),
        'umbral_buy': trial.suggest_float('xauusd_umbral_buy', 0.10, 0.25),
        'umbral_sell': trial.suggest_float('xauusd_umbral_sell', 0.10, 0.25),
    }
    # BTCUSD params
    params_btc = {
        'sl_mult': trial.suggest_float('btc_sl_mult', 0.8, 3.0),
        'tp_rr': trial.suggest_float('btc_tp_rr', 0.5, 5.0),
        'umbral_buy': trial.suggest_float('btc_umbral_buy', 0.20, 0.35),
        'umbral_sell': trial.suggest_float('btc_umbral_sell', 0.20, 0.35),
    }

    # Ejecutar cada objetivo con los parámetros sugeridos
    result_nas100 = nas100_objective(trial, custom_params=params_nas100)
    result_xauusd = xauusd_objective(trial, custom_params=params_xauusd)
    result_btc = btc_objective(trial, custom_params=params_btc)

    # Métricas clave
    profit = result_nas100.profit + result_xauusd.profit + result_btc.profit
    winrate = (result_nas100.win_rate + result_xauusd.win_rate + result_btc.win_rate) / 3
    max_dd = max(result_nas100.max_drawdown_pct, result_xauusd.max_drawdown_pct, result_btc.max_drawdown_pct)
    sharpe = (getattr(result_nas100, 'sharpe', getattr(result_nas100, 'sharpe_ratio', 0.0)) +
              getattr(result_xauusd, 'sharpe', getattr(result_xauusd, 'sharpe_ratio', 0.0)) +
              getattr(result_btc, 'sharpe', getattr(result_btc, 'sharpe_ratio', 0.0))) / 3

    # Penalización por drawdown alto
    penalty = 0.0
    if max_dd > 20:
        penalty += (max_dd - 20) * 2

    # Objetivo compuesto: profit + winrate + sharpe - penalty
    score = profit + (winrate * 10) + (sharpe * 20) - penalty

    # Logging estructurado
    log_entry = {
        'trial': trial.number,
        'params_nas100': params_nas100,
        'params_xauusd': params_xauusd,
        'params_btc': params_btc,
        'profit': profit,
        'winrate': winrate,
        'max_drawdown_pct': max_dd,
        'sharpe': sharpe,
        'score': score,
        'profit_nas100': result_nas100.profit,
        'profit_xauusd': result_xauusd.profit,
        'profit_btc': result_btc.profit,
        'winrate_nas100': result_nas100.win_rate,
        'winrate_xauusd': result_xauusd.win_rate,
        'winrate_btc': result_btc.win_rate,
        'max_dd_nas100': result_nas100.max_drawdown_pct,
        'max_dd_xauusd': result_xauusd.max_drawdown_pct,
        'max_dd_btc': result_btc.max_drawdown_pct,
        'sharpe_nas100': getattr(result_nas100, 'sharpe', getattr(result_nas100, 'sharpe_ratio', 0.0)),
        'sharpe_xauusd': getattr(result_xauusd, 'sharpe', getattr(result_xauusd, 'sharpe_ratio', 0.0)),
        'sharpe_btc': getattr(result_btc, 'sharpe', getattr(result_btc, 'sharpe_ratio', 0.0)),
        'timestamp': datetime.now().isoformat(),
    }
    logger.info(json.dumps(log_entry))
    return score

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(multiasset_objective, n_trials=50, n_jobs=1)
    # Guardar mejores parámetros
    best = study.best_trial
    with open("logs/optuna_multiasset_best.json", "w") as f:
        json.dump(best.params, f, indent=2)
    print("Mejores parámetros:", best.params)
    print("Mejor score:", best.value)

if __name__ == "__main__":
    main()
