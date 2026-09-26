#!/usr/bin/env python3
"""
Phoenix Quick Test - Valida que todo esté funcionando correctamente
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib

# Imports
try:
    from phoenix_processor import PhoenixDataProcessor
    from phoenix_metrics import TradingMetrics
    import phoenix_config as config
    print("✅ Importaciones exitosas")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

def test_config():
    """Verifica que config esté bien"""
    print(f"\n📋 TEST: Configuración")
    print(f"   Features: {config.FEATURES}")
    print(f"   Input Size: {config.INPUT_SIZE}")
    print(f"   Expected: 7 features")
    
    if config.INPUT_SIZE == 7 and len(config.FEATURES) == 7:
        print(f"   ✅ Config válida")
        return True
    else:
        print(f"   ❌ Config inválida")
        return False

def test_data_loading():
    """Verifica carga de datos"""
    print(f"\n📊 TEST: Carga de datos")
    
    if not os.path.exists(config.DATA_RAW):
        print(f"   ❌ Archivo {config.DATA_RAW} no encontrado")
        return False
    
    try:
        processor = PhoenixDataProcessor(config.DATA_RAW)
        df = processor.clean_and_prepare()
        print(f"   ✅ Datos cargados: {len(df)} velas")
        print(f"   ✅ Columnas: {list(df.columns[:10])}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_data_integrity():
    """Verifica integridad de datos"""
    print(f"\n🔍 TEST: Integridad de datos")
    
    try:
        processor = PhoenixDataProcessor(config.DATA_RAW)
        df = processor.clean_and_prepare()
        
        # Verificar features
        missing_features = [f for f in config.FEATURES if f not in df.columns]
        if missing_features:
            print(f"   ❌ Faltan features: {missing_features}")
            return False
        
        # Verificar Target
        if 'Target' not in df.columns:
            print(f"   ❌ Falta columna Target")
            return False
        
        # Verificar valores nulos
        nulls = df[config.FEATURES].isnull().sum().sum()
        if nulls > 0:
            print(f"   ⚠️  {nulls} valores nulos en features")
        
        print(f"   ✅ Features presentes: {config.FEATURES}")
        print(f"   ✅ Target distribution: {df['Target'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_model():
    """Verifica que el modelo existe y puede cargarse"""
    print(f"\n🤖 TEST: Modelo")
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"   ⚠️  Modelo no encontrado en {config.MODEL_SAVE_PATH}")
        print(f"   (Ejecuta primero: python3 phoenix_evolution.py)")
        return False
    
    try:
        if config.MODEL_TYPE == "xgboost":
            model = joblib.load(config.MODEL_SAVE_PATH)
            print(f"   ✅ Modelo XGBoost cargado correctamente")
            return True
    try:
        if config.MODEL_TYPE == "xgboost":
            model = joblib.load(config.MODEL_SAVE_PATH)
            dummy_input = np.zeros((1, config.LOOKBACK_WINDOW * config.INPUT_SIZE))
            probs = model.predict_proba(dummy_input)
            print(f"   ✅ Modelo XGBoost cargado correctamente")
            print(f"   ✅ Predict proba OK: output {probs.shape}")
            return True

        from phoenix_brain import PhoenixLSTM
        device = torch.device("cpu")
        model = PhoenixLSTM(input_size=config.INPUT_SIZE, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"   ✅ Modelo cargado correctamente")

        # Test forward pass con batch_size > 1 (para BatchNorm)
        dummy_input = torch.randn(2, config.LOOKBACK_WINDOW, config.INPUT_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   ✅ Forward pass OK: input {dummy_input.shape} -> output {output.shape}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_scaler():
    """Verifica que el scaler existe"""
    print(f"\n📏 TEST: Scaler")
    
    if not os.path.exists(config.SCALER_SAVE_PATH):
        print(f"   ⚠️  Scaler no encontrado en {config.SCALER_SAVE_PATH}")
        print(f"   (Ejecuta primero: python3 phoenix_evolution.py)")
        return False
    
    try:
        scaler = joblib.load(config.SCALER_SAVE_PATH)
        print(f"   ✅ Scaler cargado")
        print(f"   ✅ Mean: {scaler.mean_[:3]}...")
        print(f"   ✅ Scale: {scaler.scale_[:3]}...")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_metrics():
    """Verifica módulo de métricas"""
    print(f"\n📈 TEST: Módulo de Métricas")
    
    try:
        # Crear trades fake
        trades = [
            {'entry_price': 100, 'exit_price': 105, 'direction': 'BUY', 'size': 0.01, 'pnl': 5},
            {'entry_price': 105, 'exit_price': 103, 'direction': 'SELL', 'size': 0.01, 'pnl': -2},
            {'entry_price': 103, 'exit_price': 110, 'direction': 'BUY', 'size': 0.01, 'pnl': 7},
        ]
        
        metrics = TradingMetrics(200.0, trades)
        capital_history = np.array([200, 205, 203, 210])
        reporte = metrics.generar_reporte_completo(capital_history)
        
        print(f"   ✅ Métricas calculadas:")
        print(f"      - Win Rate: {reporte['win_rate']:.2f}%")
        print(f"      - Profit Factor: {reporte['profit_factor']:.2f}")
        print(f"      - Sharpe Ratio: {reporte['sharpe_ratio']:.2f}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_gpu():
    """Verifica GPU/MPS"""
    print(f"\n⚡ TEST: GPU/Hardware")
    
    if torch.backends.mps.is_available():
        print(f"   ✅ Apple Metal Performance Shaders (MPS) disponible")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print(f"   ✅ CUDA disponible")
        device = torch.device("cuda")
    else:
        print(f"   ⚠️  CPU mode (no GPU detectada)")
        device = torch.device("cpu")
    
    print(f"   Device: {device}")
    return True

def main():
    print(f"\n{'='*60}")
    print(f" PHOENIX QUICK TEST - Validación del Sistema")
    print(f"{'='*60}")
    
    tests = [
        ("Config", test_config),
        ("Data Loading", test_data_loading),
        ("Data Integrity", test_data_integrity),
        ("GPU/Hardware", test_gpu),
        ("Metrics Module", test_metrics),
        ("Scaler", test_scaler),
        ("Model", test_model),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ Exception in {name}: {e}")
            results[name] = False
    
    print(f"\n{'='*60}")
    print(f" RESUMEN")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅" if result else "⚠️ " if result is None else "❌"
        print(f"{status} {name}")
    
    print(f"\n{passed}/{total} tests pasaron")
    
    if passed == total:
        print(f"\n🎉 ¡Sistema listo para usar!")
        print(f"\nPróximos pasos:")
        print(f"1. python3 phoenix_evolution.py")
        print(f"2. python3 phoenix_parameter_optimizer.py")
        print(f"3. python3 phoenix_backtester_pro.py")
    else:
        print(f"\n⚠️  Algunos tests fallaron. Revisa los errores arriba.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
