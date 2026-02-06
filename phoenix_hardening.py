import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import warnings
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM, preparar_datos_sin_fuga
import phoenix_config as config

# --- CONFIGURACIÓN DE REHABILITACIÓN ---
# Usamos un Learning Rate muy bajo para "pulir" sin romper
LR_HARDENING = 0.00001  
EPOCHS_HARDENING = 30   
BATCH_SIZE = 4096

# FORZAR M4
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

# --- LA HERRAMIENTA DE CORRECCIÓN: FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        # Castigo exponencial a los errores en zonas de confianza
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

def iniciar_rehabilitacion():
    # Detectar M4
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"--- 🏥 INICIANDO PROTOCOLO DE REHABILITACIÓN (M4) ---")
    else:
        device = torch.device("cpu")
        print(f"--- 🏥 INICIANDO PROTOCOLO DE REHABILITACIÓN (CPU) ---")
    
    # 1. Cargar el Cerebro "Enfermo" (Overfitted)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ [ERROR] No existe el modelo para arreglar.")
        return

    print("   > Cargando paciente...")
    # Inicializamos arquitectura
    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    # Cargamos pesos (mapeo seguro para mps/cpu)
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    
    # 2. Cargar Datos (Los mismos, para que re-aprenda a verlos bien)
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    # Usamos la función del brain para obtener tensores
    X_train, y_train, _, _ = preparar_datos_sin_fuga(df)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0 
    )

    # 3. Configurar la "Cirugía" 
    # Optimizador lento + Focal Loss agresivo (Gamma=3)
    optimizer = optim.Adam(model.parameters(), lr=LR_HARDENING)
    criterion = FocalLoss(gamma=3.0) 

    print(f"--- [HARDENING] Aplicando corrección de sesgo a {len(X_train)} patrones ---")
    print(f"--- Objetivo: Reducir falsos positivos (Win Rate actual: 20%) ---")
    
    model.train()
    for epoch in range(EPOCHS_HARDENING):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Tratamiento [{epoch+1}/{EPOCHS_HARDENING}] - Loss Correctiva: {total_loss/len(train_loader):.6f}")

    # 4. Guardar
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"--- [EXITO] Cerebro rehabilitado guardado en {config.MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    iniciar_rehabilitacion()