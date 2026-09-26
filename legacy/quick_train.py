#!/usr/bin/env python3
"""
Phoenix Quick Train - Versión rápida para demo (30 segundos)
Solo 3 épocas para validar que funciona
"""

import phoenix_config as config

if config.MODEL_TYPE == "xgboost":
    raise SystemExit("Quick train LSTM no aplica en modo XGBoost.")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import PhoenixLSTM

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

# 1. Cargar datos
print("\n📊 Cargando datos...")
processor = PhoenixDataProcessor(config.DATA_RAW)
df = processor.clean_and_prepare()
print(f"✅ {len(df)} velas")

# 2. Preparar datos
print("\n🔧 Preparando secuencias...")
features = config.FEATURES
split_idx = int(len(df) * 0.7)
data = df[features].values
target = df['Target'].values

# Split ANTES de escalar
data_train = data[:split_idx]
target_train = target[:split_idx]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_train)
data_train_scaled = scaler.transform(data_train)

# Crear secuencias
X_train, y_train = [], []
for i in range(config.LOOKBACK_WINDOW, len(data_train_scaled)):
    X_train.append(data_train_scaled[i-config.LOOKBACK_WINDOW:i])
    y_train.append(target_train[i])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.long)

print(f"✅ X_train shape: {X_train.shape}")

# 3. Crear loader
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)

# 4. Crear modelo
print("\n🧠 Inicializando modelo...")
model = PhoenixLSTM(
    input_size=config.INPUT_SIZE,
    hidden_layers=config.HIDDEN_LAYERS,
    num_classes=3,
    dropout=config.DROPOUT_RATE
).to(DEVICE)

classes, counts = np.unique(y_train.numpy(), return_counts=True)
weights = 1.0 / counts
weights_tensor = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)

# 5. Entrenar (RÁPIDO - 3 épocas)
print(f"\n🚀 Entrenando (3 épocas - DEMO)...")
model.train()
for epoch in range(3):
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"   Época {epoch+1}/3 | Loss: {avg_loss:.4f}")

# 6. Guardar
print(f"\n✅ Guardando modelo y scaler...")
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
joblib.dump(scaler, config.SCALER_SAVE_PATH)

print(f"✅ phoenix_brain.pth guardado")
print(f"✅ phoenix_scaler.pkl guardado")
print(f"\n✅ Demo entrenamiento completado!")
print(f"Ahora ejecuta: python3 test_system.py")
