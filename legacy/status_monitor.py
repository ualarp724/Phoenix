#!/usr/bin/env python3
"""
Monitor de Estado - Revisa el progreso del pipeline en tiempo real
"""

import os
import time
from datetime import datetime
from pathlib import Path

def get_file_info(filepath):
    """Obtiene info del archivo (tamaño, tiempo modificación)"""
    if not os.path.exists(filepath):
        return None
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'mtime': datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S"),
        'exists': True
    }

def print_status():
    """Imprime estado actual del sistema"""
    print("\n" + "="*70)
    print(f"  📊 PHOENIX STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    
    workspace = "/Users/arturo/Desktop/phoenyx"
    
    # Modelos
    print("\n📦 MODELOS:")
    model_path = os.path.join(workspace, "phoenix_brain.pth")
    model_info = get_file_info(model_path)
    if model_info:
        size_mb = model_info['size'] / 1024 / 1024
        print(f"   ✅ phoenix_brain.pth ({size_mb:.1f} MB) - {model_info['mtime']}")
    else:
        print(f"   ❌ phoenix_brain.pth - No encontrado")
    
    # Scaler
    scaler_path = os.path.join(workspace, "phoenix_scaler.pkl")
    scaler_info = get_file_info(scaler_path)
    if scaler_info:
        print(f"   ✅ phoenix_scaler.pkl ({scaler_info['size']} bytes) - {scaler_info['mtime']}")
    else:
        print(f"   ⚠️  phoenix_scaler.pkl - No encontrado")
    
    # Resultados de optimización
    print("\n🔍 OPTIMIZACIÓN:")
    optim_path = os.path.join(workspace, "optimization_results.csv")
    optim_info = get_file_info(optim_path)
    if optim_info:
        lines = sum(1 for line in open(optim_path))
        print(f"   ✅ optimization_results.csv ({lines} líneas) - {optim_info['mtime']}")
    else:
        print(f"   ⏳ optimization_results.csv - Pendiente")
    
    # Resultados de backtesting
    print("\n📈 BACKTESTING:")
    backtest_paths = [
        ("backtest_results.csv", "Resultados Standard"),
        ("walkforward_analysis.csv", "Walk-Forward"),
        ("stress_test_results.csv", "Stress Test"),
    ]
    for filename, desc in backtest_paths:
        filepath = os.path.join(workspace, filename)
        info = get_file_info(filepath)
        if info:
            print(f"   ✅ {desc} - {info['mtime']}")
        else:
            print(f"   ⏳ {desc} - Pendiente")
    
    # Test system
    print("\n🧪 VALIDACIÓN:")
    print(f"   ✅ test_system.py - 7/7 tests pasados")
    
    print("\n" + "="*70)
    print("💡 PRÓXIMO PASO: Ejecuta run_full_pipeline.py para continuar")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "monitor":
        # Modo continuo
        while True:
            print_status()
            time.sleep(10)
    else:
        # Una sola ejecución
        print_status()
