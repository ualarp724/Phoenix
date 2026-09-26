#!/usr/bin/env python3
"""
Phoenix Auto-Runner - Ejecuta automáticamente el siguiente paso cuando el anterior termina
Monitorea la última modificación de phoenix_brain.pth para saber cuándo termina el entrenamiento
"""

import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def wait_for_training_completion():
    """Espera a que phoenix_evolution.py termine (monitoreando el archivo)"""
    model_path = "phoenix_brain.pth"
    
    print("⏳ Esperando que el entrenamiento termine...")
    print(f"   Monitoreando: {model_path}")
    print()
    
    # Obtener tiempo inicial
    last_mtime = os.path.getmtime(model_path)
    stable_count = 0
    required_stable = 5  # 5 chequeos sin cambios = probablemente terminó
    
    while stable_count < required_stable:
        time.sleep(10)  # Chequear cada 10 segundos
        
        current_mtime = os.path.getmtime(model_path)
        current_size = os.path.getsize(model_path)
        
        if current_mtime == last_mtime:
            stable_count += 1
            status = "━" * stable_count + "○ " * (required_stable - stable_count)
        else:
            last_mtime = current_mtime
            stable_count = 0
            status = "🔄 Actualizándose... "
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\r[{timestamp}] {status} ({current_size/1024/1024:.1f} MB)", end="")
        sys.stdout.flush()
    
    print("\n\n✅ ¡Entrenamiento completado!")
    return True

def run_optimizer():
    """Ejecuta el parameter optimizer"""
    print("\n" + "="*70)
    print("  🔍 INICIANDO: Parameter Optimizer (96 combinaciones)")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run([sys.executable, "phoenix_parameter_optimizer_v2.py"])
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_backtester():
    """Ejecuta el professional backtester"""
    print("\n" + "="*70)
    print("  📈 INICIANDO: Professional Backtester")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run([sys.executable, "phoenix_backtester_pro.py"])
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  🚀 PHOENIX AUTO-RUNNER")
    print("="*70)
    print()
    
    # Verificar que el entrenamiento está en curso
    if not os.path.exists("phoenix_brain.pth"):
        print("❌ No se encontró phoenix_brain.pth")
        print("   Primero ejecuta: python3 phoenix_evolution.py")
        return
    
    # Esperar a que termine el entrenamiento
    if not wait_for_training_completion():
        print("❌ Error esperando entrenamiento")
        return
    
    # Ejecutar optimizer
    if not run_optimizer():
        print("❌ Error en optimizer")
        return
    
    print("\n📊 Revisa optimization_results.csv para ver TOP 5 parámetros")
    print("    Actualiza UMBRAL_CONFIANZA, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER en phoenix_config.py")
    
    # Pregunta al usuario
    response = input("\n¿Ejecutar backtester? (s/n): ").strip().lower()
    if response == 's':
        if not run_backtester():
            print("❌ Error en backtester")
            return
    
    print("\n" + "="*70)
    print("  ✅ Pipeline completado exitosamente")
    print("="*70)
    print("\n🎯 Próximo paso: python3 phoenix_live.py")

if __name__ == "__main__":
    main()
