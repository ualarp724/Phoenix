import torch
import onnx
import os
from phoenix_brain import PhoenixLSTM
import phoenix_config as config

def exportar_precision():
    print("--- [SISTEMA] Exportando Cerebro de Precisión (Monolítico) ---")
    model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3)
    
    # Cargamos el archivo de endurecimiento
    model.load_state_dict(torch.load("phoenix_brain_precision.pth", map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, config.LOOKBACK_WINDOW, 6)
    temp_name = "temp_precision.onnx"
    final_name = "phoenix_brain.onnx"

    # Exportación
    torch.onnx.export(model, dummy_input, temp_name, export_params=True, opset_version=17)

    # Fusión Monolítica
    onnx_model = onnx.load(temp_name)
    onnx.save_model(onnx_model, final_name, save_as_external_data=False)
    
    size_kb = os.path.getsize(final_name) / 1024
    print(f"--- [EXITO] Archivo: {final_name} | Tamaño: {size_kb:.2f} KB ---")
    if os.path.exists(temp_name): os.remove(temp_name)

if __name__ == "__main__":
    exportar_precision()