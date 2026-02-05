import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import warnings
import os
from phoenix_processor import PhoenixDataProcessor
import phoenix_config as config

# 1. SILENCIAR ADVERTENCIAS DE INFRAESTRUCTURA
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# 2. DEFINICIÓN DE ARQUITECTURA
class PhoenixLSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(PhoenixLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layers[0], batch_first=True, num_layers=3, dropout=0.3)
        self.fc = nn.Linear(hidden_layers[0], num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def preparar_secuencias(df):
    features = ['Open', 'High', 'Low', 'Close', 'Returns', 'Z_Score_Vol']
    data = df[features].values
    target = df['Target'].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    lookback = config.LOOKBACK_WINDOW
    # Vectorización de alta velocidad
    X = np.array([data_scaled[i : i + lookback] for i in range(len(data_scaled) - lookback)])
    y = target[lookback:]
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), scaler

def entrenar():
    # Detectar dispositivo una sola vez
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Logs iniciales de administración
    print(f"--- [EJECUCIÓN] Iniciando Motor Phoenix | Dispositivo: {device} ---")
    
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    X, y, scaler = preparar_secuencias(df)
    
    split = int(len(X) * 0.8)
    dataset_train = TensorDataset(X[:split], y[:split])

    # OPTIMIZACIÓN MÁXIMA: Batch Size de 2048 para estrujar la RAM de 16GB
    loader = DataLoader(
        dataset_train, 
        batch_size=2048, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True # Mantiene los hilos vivos para evitar latencia
    )

    model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"--- [TRAINING] Procesando {len(X)} secuencias de Oro ---")
    
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Output limpio de alta dirección
        if (epoch + 1) % 5 == 0:
            print(f"Época [{epoch+1}/{config.EPOCHS}] - Loss: {total_loss/len(loader):.4f}")

    # Guardado de activos
    torch.save(model.to("cpu").state_dict(), config.MODEL_SAVE_PATH)
    joblib.dump(scaler, "phoenix_scaler.pkl")
    print(f"--- [EXITO] Modelo consolidado en {config.MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    entrenar()