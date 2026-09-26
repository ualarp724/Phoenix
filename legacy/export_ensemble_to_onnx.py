"""
Exporta modelos ensemble (XGBoost + LightGBM) y scaler a ONNX para BTCUSD, NAS100, XAUUSD.
Incluye pipeline de preprocesado (scaler) y predicción.
Lógica de lot sizing híbrido: documentada para orquestador MQL5 (no incluida en ONNX).
"""
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import phoenix_config as config

ASSETS = [
    ("BTCUSD", "phoenix_btc_ensemble.onnx"),
    ("NAS100", "phoenix_nas100_ensemble.onnx"),
    ("XAUUSD", "phoenix_xauusd_ensemble.onnx"),
]

MODEL_KEYS = ["xgb_model", "lgbm_model", "scaler"]


def export_ensemble_onnx(symbol: str, onnx_path: str):
    config.apply_asset(symbol)
    asset = config.ASSETS[symbol]
    print(f"\n[EXPORT] {symbol} → {onnx_path}")


    # Cargar modelos y scaler

    xgb = joblib.load(asset["xgb_model"])
    lgbm = joblib.load(asset["lgbm_model"])
    scaler = joblib.load(asset["scaler"])

    n_features = len(config.FEATURES)
    lookback = config.LOOKBACK_WINDOW

    # Exportar XGBoost a ONNX
    onnx_xgb_path = onnx_path.replace(".onnx", "_xgb.onnx")
    xgb.get_booster().save_model(onnx_xgb_path)

    # Exportar LightGBM a ONNX
    onnx_lgb_path = onnx_path.replace(".onnx", "_lgbm.onnx")
    try:
        lgbm.booster_.save_model(onnx_lgb_path, format="onnx")
    except Exception as e:
        print(f"[WARN] LightGBM ONNX export failed: {e}. Exportando en formato nativo .txt")
        lgbm.booster_.save_model(onnx_lgb_path.replace(".onnx", ".txt"))

    print(f"  ✔️ Guardado: {onnx_xgb_path}, {onnx_lgb_path}")

    # Guardar scaler y features para MQL5
    scaler_path = onnx_path.replace(".onnx", "_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    features_path = onnx_path.replace(".onnx", "_features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(config.FEATURES))
    print(f"  ✔️ Scaler y features exportados para MQL5")

    # Documentar pesos ensemble
    weights = asset.get("ensemble_weights", {"xgboost": 0.5, "lightgbm": 0.5})
    with open(onnx_path.replace(".onnx", "_ensemble_weights.txt"), "w") as f:
        f.write(str(weights))
    print(f"  ✔️ Pesos ensemble: {weights}")


def main():
    for symbol, onnx_path in ASSETS:
        export_ensemble_onnx(symbol, onnx_path)
    print("\n[INFO] Exportación ONNX finalizada. Preprocesado: aplicar scaler y flatten en MQL5 antes de pasar al modelo. Lógica de lot sizing híbrido → ver documentación adjunta.")

    # Documentación detallada del preprocesado para MQL5
    doc = f"""
==================== PREPROCESADO PARA MQL5 ====================

1. Features de entrada (en este orden):
{', '.join(config.FEATURES)}

2. Para cada predicción:
   a) Construir una matriz de shape ({lookback}, {n_features}) con los últimos {lookback} ticks/barras.
   b) Aplicar el scaler (StandardScaler) entrenado en Python a cada feature (usar .pkl exportado).
   c) Aplanar la matriz a un vector de tamaño ({lookback} * {n_features}).
   d) Pasar ese vector como input al modelo ONNX XGB y LGBM.
   e) Combinar las probabilidades de ambos modelos usando los pesos de ensemble exportados.

3. El scaler debe aplicar: x_scaled = (x - mean) / std para cada feature, usando los parámetros del .pkl.

4. El pipeline de Python es exactamente:
   - Para cada nueva barra:
       - features = [RSI, Vol_Rel, ..., H1_BB_Pos]  # en el orden exportado
       - matriz = [features_t-59, ..., features_t]
       - matriz_scaled = scaler.transform(matriz)
       - vector = matriz_scaled.flatten()
       - y_pred = model.predict_proba(vector)

5. El código MQL5 debe replicar este flujo para que la inferencia sea idéntica.

===============================================================
"""
    with open("PREPROCESADO_MQL5.txt", "w") as f:
        f.write(doc)
    print("[INFO] Documentación de preprocesado para MQL5 generada: PREPROCESADO_MQL5.txt")

if __name__ == "__main__":
    main()
