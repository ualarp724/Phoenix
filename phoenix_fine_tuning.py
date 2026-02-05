import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM, preparar_secuencias
import phoenix_config as config

# 1. CONFIGURACIÓN DE PRECISIÓN
LR_FINE_TUNING = 0.0001 # 10 veces más pequeño que el original
EPOCHS_FT = 50
BATCH_SIZE_FT = 2048

def iniciar_perfeccionamiento():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- [EJECUCIÓN] Perfeccionando Cerebro Phoenix | Dispositivo: {device} ---")
    
    # Cargar Datos
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    X, y, _ = preparar_secuencias(df) # Usamos el scaler que ya tenemos
    
    split = int(len(X) * 0.8)
    dataset_train = TensorDataset(X[:split], y[:split])
    loader = DataLoader(dataset_train, batch_size=BATCH_SIZE_FT, shuffle=True, num_workers=4)

    # CARGAR MODELO EXISTENTE (EL QUE GANÓ $5,047)
    model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print("--- [INFO] Inteligencia previa cargada correctamente ---")
    except:
        print("[ERROR] No se encontró el modelo previo. Ejecute phoenix_brain.py primero.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_FINE_TUNING) # Tasa de aprendizaje de precisión

    print(f"--- [FINE-TUNING] Puliendo pesos durante {EPOCHS_FT} épocas ---")
    
    model.train()
    for epoch in range(EPOCHS_FT):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Refinamiento [{epoch+1}/{EPOCHS_FT}] - Loss: {total_loss/len(loader):.4f}")

    # Guardar la nueva versión perfeccionada
    torch.save(model.to("cpu").state_dict(), config.MODEL_SAVE_PATH)
    print(f"--- [EXITO] Cerebro perfeccionado y guardado ---")

if __name__ == "__main__":
    iniciar_perfeccionamiento()