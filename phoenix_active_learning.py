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

# CONFIGURACIÓN DE APRENDIZAJE CONTINUO
LR_FINE_TUNING = 0.00005  # 20 veces más lento que el entrenamiento normal (Cirugía de precisión)
EPOCHS_ACTIVE = 30        # Pocas épocas para no sobreajustar
BATCH_SIZE = 1024         # Lotes más pequeños para generalizar mejor

# --- HARDENING: FOCAL LOSS ---
# Esta función penaliza los errores en predicciones "seguras".
# Obliga a la IA a no confiarse y buscar el 3% con certeza matemática.
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

def iniciar_auto_mejora():
    # Detectar M4
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- [AUTONOMOUS LEARNING] Iniciando Protocolo de Mejora Continua | {device} ---")
    
    # 1. Cargar Cerebro Existente
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("❌ [ERROR] No existe un cerebro base. Ejecuta phoenix_brain.py primero.")
        return

    print("   > Cargando conocimientos previos...")
    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # 2. Cargar Nuevos Datos (Simulación de "Lo que pasó esta semana")
    # Usamos 'vantage_live_gold.csv' como la fuente de nuevos datos
    archivo_nuevos_datos = "vantage_live_gold.csv" 
    if not os.path.exists(archivo_nuevos_datos):
        print(f"⚠️ No encuentro {archivo_nuevos_datos}. Usando datos base para demostración de Hardening.")
        archivo_nuevos_datos = config.DATA_RAW

    processor = PhoenixDataProcessor(archivo_nuevos_datos)
    df = processor.clean_and_prepare()
    
    # IMPORTANTE: Usamos el Scaler original para que la IA entienda los datos igual que antes
    # (No re-entrenamos el scaler, solo transformamos)
    # Nota: Aquí reutilizamos la lógica de brain pero asumiendo que el scaler existe
    X_train, y_train, _, _ = preparar_datos_sin_fuga(df) 
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # 3. Configurar la "Cirugía" (Optimizador Lento + Focal Loss)
    optimizer = optim.Adam(model.parameters(), lr=LR_FINE_TUNING)
    criterion = FocalLoss(gamma=2.5) # Gamma alto = Exigencia alta

    print(f"--- [HARDENING] Refinando estrategia con {len(X_train)} nuevas velas ---")
    
    model.train()
    for epoch in range(EPOCHS_ACTIVE):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Mejora [{epoch+1}/{EPOCHS_ACTIVE}] - Loss (Focal): {total_loss/len(train_loader):.6f}")

    # 4. Guardar la versión mejorada (Sobrescribe la anterior)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"--- [EXITO] Conocimiento integrado. El bot es ahora más inteligente. ---")

if __name__ == "__main__":
    iniciar_auto_mejora()