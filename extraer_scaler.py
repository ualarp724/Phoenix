import joblib
import numpy as np

try:
    scaler = joblib.load("phoenix_scaler.pkl")
    print("\n" + "="*40)
    print(" DATOS DEL SCALER PARA MQL5")
    print("="*40)
    means = ", ".join([f"{x:.8f}" for x in scaler.mean_])
    scales = ", ".join([f"{x:.8f}" for x in scaler.scale_])
    
    print(f"double scaler_means[] = {{{means}}};")
    print(f"double scaler_scales[] = {{{scales}}};")
    print("="*40)
except Exception as e:
    print(f"Error: {e}")