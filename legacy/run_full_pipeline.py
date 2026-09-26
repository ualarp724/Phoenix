#!/usr/bin/env python3
"""
Pipeline completo Phoenix - Ejecuta en secuencia:
1. Entrenamiento del modelo (phoenix_evolution.py)
2. Optimización de parámetros (96 combinaciones)
3. Backtesting profesional (walk-forward + stress test)
"""

import os
import subprocess
import sys
from datetime import datetime

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def run_script(script_name, description):
    """Ejecuta un script Python y espera a que termine"""
    log(f"{'='*60}")
    log(f"▶️  INICIANDO: {description}")
    log(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], cwd="/Users/arturo/Desktop/phoenyx")
        if result.returncode == 0:
            log(f"✅ {description} completado")
            return True
        else:
            log(f"❌ {description} falló con código {result.returncode}")
            return False
    except Exception as e:
        log(f"❌ Error ejecutando {script_name}: {e}")
        return False

def main():
    log(f"\n🚀 PHOENIX FULL PIPELINE INICIADO")
    log(f"Workspace: /Users/arturo/Desktop/phoenyx")
    
    steps = [
        ("phoenix_evolution.py", "ENTRENAMIENTO DEL MODELO (100 épocas)"),
        ("phoenix_parameter_optimizer.py", "OPTIMIZACIÓN DE PARÁMETROS (96 combos)"),
        ("phoenix_backtester_pro.py", "BACKTESTING PROFESIONAL (Walk-Forward)"),
    ]
    
    completed = 0
    for script, desc in steps:
        if run_script(script, desc):
            completed += 1
        else:
            log(f"⚠️  Pipeline detenido en: {desc}")
            break
        log("")
    
    log(f"{'='*60}")
    log(f"📊 RESUMEN: {completed}/{len(steps)} pasos completados")
    log(f"{'='*60}")
    
    if completed == len(steps):
        log(f"\n🎉 ¡Pipeline completo exitoso!")
        log(f"Próximos pasos:")
        log(f"1. Revisa optimization_results.csv para ver los TOP 5 parámetros")
        log(f"2. Actualiza UMBRAL_CONFIANZA, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER en phoenix_config.py")
        log(f"3. Ejecuta: python3 phoenix_live.py para trading en vivo")
    else:
        log(f"\n⚠️  El pipeline se detuvo. Verifica los errores arriba.")

if __name__ == "__main__":
    main()
