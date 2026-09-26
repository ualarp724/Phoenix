# 🚀 PHOENIX BOT - GUÍA DE INICIO RÁPIDO

## OPCIÓN 1: Ejecución Automática (Recomendado)

```bash
cd /Users/arturo/Desktop/phoenyx
python3 run_pipeline.py
```

Esto ejecuta automáticamente:
1. ✅ Validación del sistema
2. 🧠 Entrenamiento del modelo
3. 🔍 Optimización de parámetros
4. 📊 Backtesting profesional
5. 📈 Generación de reporte

**Tiempo estimado**: 10-30 minutos (depende de tu CPU/GPU)

---

## OPCIÓN 2: Ejecución Manual Paso a Paso

### PASO 1: Validar Sistema
```bash
python3 test_system.py
```

**Qué hace**: Verifica que todos los archivos, features y dependencias estén ok.

**Output esperado**:
```
✅ Importaciones exitosas
✅ Config válida
✅ Datos cargados: 2500 velas
✅ Features presentes
✅ Apple Metal Performance Shaders disponible
✅ Módulo de Métricas OK

🎉 ¡Sistema listo para usar!
```

---

### PASO 2: Entrenar Modelo
```bash
python3 phoenix_evolution.py
```

**Qué hace**: 
- Carga datos sin data leakage
- Entrena LSTM con validación cruzada
- Usa early stopping automático
- Guarda modelo y scaler

**Output esperado**:
```
Datos cargados: 2500 velas
Target distribution: {0: 1800, 1: 450, 2: 250}

Preparando secuencias...
✅ Scaler guardado en phoenix_scaler.pkl

Inicializando modelo...
⚖️  Pesos de Clases: {0: 0.15, 1: 0.65, 2: 1.05}

Entrenando modelo...
Época  10/100 | Train Loss: 0.8234 | Val Loss: 0.8156 | Val Acc: 62.5%
Época  20/100 | Train Loss: 0.7456 | Val Loss: 0.7823 | Val Acc: 65.2%
...
Época  45/100 | Early Stopping (paciencia agotada)
✅ Modelo entrenado guardado en phoenix_brain.pth

======================================================================
 RESULTADOS DEL ENTRENAMIENTO
======================================================================
Capital Final: $247.50
Retorno: 23.75%
Total Operaciones: 8
Win Rate: 62.5%
Sharpe: 1.45 | Calmar: 0.92
======================================================================
```

**Archivos generados**:
- `phoenix_brain.pth` - Modelo entrenado
- `phoenix_scaler.pkl` - Scaler para normalizar datos

---

### PASO 3: Optimizar Parámetros
```bash
python3 phoenix_parameter_optimizer.py
```

**Qué hace**: 
- Prueba 96 combinaciones de parámetros
- Calcula Sharpe, Calmar, Win Rate para cada una
- Retorna TOP 5 mejores configuraciones
- Genera `optimization_results.csv`

**Output esperado**:
```
🔍 Evaluando 96 combinaciones...

CONF  | SL   | TP   | PROFIT    | TRADES | WR %    | PF    | SHARPE | CALMAR | SCORE
0.55  | 1.0  | 1.5  | $12.50    | 3      | 66.67%  | 1.50  | 0.82   | 0.41   | 0.54
0.55  | 1.0  | 2.0  | $18.75    | 4      | 75.00%  | 2.00  | 1.15   | 0.57   | 0.72
...

================================================================================
 TOP 5 MEJORES CONFIGURACIONES
================================================================================

1. UMBRAL=0.68 | SL=1.5 | TP=2.0
   Capital: $247.50 | Profit: $47.50 (23.75%)
   Operaciones: 8 | Win Rate: 62.5%
   Sharpe: 1.85 | Calmar: 0.92 | Score: 0.84

2. UMBRAL=0.70 | SL=1.5 | TP=2.5
   Capital: $235.20 | Profit: $35.20 (17.60%)
   Operaciones: 6 | Win Rate: 66.67%
   Sharpe: 1.72 | Calmar: 0.86 | Score: 0.81

[... más configuraciones ...]

💡 RECOMENDACIÓN:
   Actualiza phoenix_config.py con:
   UMBRAL_CONFIANZA = 0.68
   ATR_SL_MULTIPLIER = 1.5
   ATR_TP_MULTIPLIER = 2.0
```

**Archivos generados**:
- `optimization_results.csv` - Todos los resultados (96 filas)

---

### PASO 4: Actualizar Configuración
```bash
# Edita phoenix_config.py
# Reemplaza:
UMBRAL_CONFIANZA = 0.68        # ← Cambiar
ATR_SL_MULTIPLIER = 1.5        # ← Cambiar
ATR_TP_MULTIPLIER = 2.0        # ← Cambiar
```

---

### PASO 5: Backtesting Profesional
```bash
python3 phoenix_backtester_pro.py
```

**Qué hace**:
- Backtesting completo con métricas detalladas
- Walk-Forward Analysis (detecta overfitting)
- Stress Testing (impacto de slippage real)

**Output esperado**:
```
======================================================================
 RESULTADOS DEL BACKTEST
======================================================================
Capital Final:         $247.50
Profit:                $47.50
Total Operaciones:     8
Win Rate:              62.5%
Profit Factor:         1.85
Sharpe Ratio:          1.85
Calmar Ratio:          0.92
======================================================================

🔄 Walk-Forward Analysis (20 velas por ventana)
VENTANA    | PROFIT     | TRADES | WIN%   | SHARPE
1          | $32.50     | 8      | 62.5%  | 1.23
2          | $18.75     | 5      | 60.0%  | 0.85
3          | $45.20     | 12     | 66.7%  | 1.52

⚡ Stress Test - Impacto de Slippage
SLIPPAGE % | PROFIT     | WIN RATE | SHARPE
0.0        | $47.50     | 62.5%    | 1.85
0.1        | $43.20     | 62.5%    | 1.72
0.2        | $38.90     | 62.5%    | 1.58
0.5        | $28.30     | 62.5%    | 1.23

✅ Backtesting completado
```

---

## PASO 6: Evaluar Métricas

### Métricas Requeridas para 3% Diario
```
✅ Win Rate > 55%              → Deberías tener > 60%
✅ Profit Factor > 1.8         → Deberías tener > 1.85
✅ Sharpe Ratio > 1.5          → Deberías tener > 1.85
✅ Calmar Ratio > 0.75         → Deberías tener > 0.92
✅ Max Drawdown < 20%          → Verificar en walk-forward
```

### Interpretación
```
Si todo está ✅:
  → El bot está listo para trading en vivo
  → Usa los parámetros optimizados
  → Monitorea diariamente

Si hay ⚠️:
  → Intenta diferentes FEATURE ENGINEERING
  → Ajusta manual de SL/TP
  → Reentren el modelo
```

---

## PASO 7: Trading en Vivo (Cuando Esté Listo)

```bash
python3 phoenix_live.py
```

**IMPORTANTE - ANTES DE USAR EN VIVO:**

1. ✅ Validar en backtest por 100+ trades
2. ✅ Win Rate > 55%
3. ✅ Sharpe > 1.5
4. ✅ Drawdown < 20% en walk-forward
5. ✅ Capital mínimo $500 (no $200)
6. ✅ Paper trading 1-2 semanas
7. ✅ Circuit breakers en phoenix_live.py

---

## 🔍 TROUBLESHOOTING

### "❌ Error de importación"
```bash
# Reinstala dependencias
pip install torch pandas numpy scikit-learn joblib

# O en tu venv:
source venv_stable/bin/activate
pip install --upgrade torch
```

### "❌ Modelo no encontrado"
```bash
# Necesitas entrenar primero
python3 phoenix_evolution.py
```

### "❌ Scaler no encontrado"
```bash
# Se genera al entrenar
python3 phoenix_evolution.py
# Genera: phoenix_scaler.pkl
```

### "⚠️ Performance baja en live vs backtest"
```
Causas comunes:
1. Slippage real > stress test
2. Spread más ancho en horas bajas liquidez
3. Cambios de mercado (reentrenar)
4. Overfitting (revisar walk-forward)

Soluciones:
1. Aumenta UMBRAL_CONFIANZA
2. Aumenta ATR_SL_MULTIPLIER
3. Sé más selectivo
4. Reoptimiza parámetros
```

---

## 📊 ESTRUCTURA DE CARPETA

```
/phoenyx/
├── phoenix_config.py                    ← Configuración central
├── phoenix_brain.py                     ← Arquitectura del modelo
├── phoenix_processor.py                 ← Procesamiento de datos
├── phoenix_evolution.py                 ← Entrenamiento MEJORADO
├── phoenix_metrics.py                   ← Métricas profesionales NUEVO
├── phoenix_parameter_optimizer.py       ← Grid search NUEVO
├── phoenix_backtester_pro.py           ← Backtesting NUEVO
├── phoenix_live.py                      ← Trading en vivo
├── test_system.py                       ← Validación NUEVO
├── run_pipeline.py                      ← Auto-pipeline NUEVO
├── MEJORAS_IMPLEMENTADAS.md             ← Documentación NUEVO
├── QUICK_REFERENCE.md                   ← Referencia rápida NUEVO
├── this_file.txt                        ← Esta guía NUEVO
│
├── phoenix_brain.pth                    ← Modelo (generado)
├── phoenix_scaler.pkl                   ← Scaler (generado)
├── optimization_results.csv             ← Resultados (generado)
│
├── vantage_gold.csv                     ← Datos históricos
├── vantage_live_gold.csv                ← Datos en vivo (MT5)
│
└── venv_stable/                         ← Virtual environment
```

---

## ⏱️ TIEMPOS ESTIMADOS

| Operación | Tiempo |
|-----------|--------|
| test_system.py | 10 segundos |
| phoenix_evolution.py | 3-5 minutos |
| parameter_optimizer.py | 10-15 minutos |
| backtester_pro.py | 5-10 minutos |
| **Total completo** | **20-35 minutos** |

Con GPU (MPS): Mitad del tiempo

---

## 💡 TIPS FINALES

1. **Reproducibilidad**: Los mismos datos + same seed = mismo resultado
2. **Paciencia**: No overwrites los parámetros. Usa los del optimizer
3. **Monitoring**: Log diario de trades y métricas
4. **Adaptación**: Si market cambia, reoptimiza (cada mes)
5. **Capital**: $200 es muy poco. Considera $500-1000 mínimo

---

## 📞 PRÓXIMAS MEJORAS POSIBLES

- [ ] Machine Learning features (momentum, trend)
- [ ] Ensemble de modelos (LSTM + GBM + SVM)
- [ ] Reinforcement Learning para position sizing
- [ ] Multi-timeframe analysis (M5 + M15 + H1)
- [ ] Portfolio mode (XAUUSD + EURUSD + GBPUSD)

---

**¡Buena suerte! Que ganes 3% diario 🚀**

---

Creado: 7 de febrero de 2026
Versión: 2.0 (Mejorado con data leakage fix, early stopping, métricas profesionales)
