#!/usr/bin/env python3
"""
Phoenix Auto-Pipeline - Ejecuta todo automáticamente
1. Test system
2. Entrena modelo
3. Optimiza parámetros
4. Backtea
5. Genera reporte
"""

import subprocess
import sys
import os
from datetime import datetime

import phoenix_config as config

def run_command(cmd, description):
    """Ejecuta un comando y retorna si fue exitoso"""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\n✅ {description} - OK")
            return True
        else:
            print(f"\n❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print(f"\n{'='*70}")
    print(f" PHOENIX AUTO-PIPELINE v1.0")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    steps = [
        ("python3 test_system.py", "1️⃣  Validar Sistema"),
        ("python3 phoenix_evolution.py", "2️⃣  Entrenar Modelo"),
    ]

    if config.MODEL_TYPE != "xgboost":
        steps.append(("python3 phoenix_parameter_optimizer.py", "3️⃣  Optimizar Parámetros"))
    else:
        print("⚠️  Optimización de parámetros omitida para XGBoost.")

    steps.append(("python3 phoenix_backtester_pro.py", "4️⃣  Backtesting Profesional"))
    
    results = {}
    for i, (cmd, desc) in enumerate(steps, 1):
        success = run_command(cmd, desc)
        results[desc] = success
        
        if not success and i <= 1:  # Si falla validación, no continúa
            print(f"\n❌ Sistema no pasó validación. Abortar.")
            break
        
        if not success and i == 2:  # Si falla entrenamiento, no continúa
            print(f"\n❌ Modelo no entrenó. Abortar.")
            break
    
    # Reporte final
    print(f"\n{'='*70}")
    print(f" REPORTE FINAL")
    print(f"{'='*70}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for desc, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {desc}")
    
    print(f"\n{passed}/{total} pasos completados correctamente\n")
    
    if passed == total:
        print(f"🎉 ¡Pipeline completado exitosamente!")
        print(f"\n📊 Próximos pasos:")
        print(f"   1. Revisa optimization_results.csv para ver TOP parámetros")
        print(f"   2. Actualiza phoenix_config.py con los mejores parámetros")
        print(f"   3. Cuando esté listo, usa phoenix_live.py para trading en vivo")
        return 0
    else:
        print(f"⚠️  Algunos pasos fallaron. Revisa los errores arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
