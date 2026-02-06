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
import time
import gc # Garbage Collector
from phoenix_processor import PhoenixDataProcessor
import phoenix_config as config

# --- CONFIGURACIÓN DE MEMORIA APPLE SILICON ---
# Permite usar toda la memoria disponible sin límites artificiales
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

# --- ARQUITECTURA NEURONAL ---
class PhoenixLSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(PhoenixLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layers[0], batch_first=True, num_layers=3, dropout=0.3)
        self.fc_1 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_layers[1], num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc_1(out)
        out = self.relu(out)
        return self.fc_2(out)

def preparar_datos_sin_fuga(df):
    future_window = 5 
    min_profit_factor = 1.0 
    future_return = df['Close'].shift(-future_window) - df['Close']
    df['Target'] = 0
    df.loc[future_return > (df['ATR'] * min_profit_factor), 'Target'] = 1 
    df.loc[future_return < -(df['ATR'] * min_profit_factor), 'Target'] = 2 
    df.dropna(inplace=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'Vol_Z', 'Dist_EMA200']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[features])
    X_test_scaled = scaler.transform(test_df[features]) 
    joblib.dump(scaler, config.SCALER_SAVE_PATH)
    
    # Vectorización
    def create_sequences_vectorized(data, targets, seq_length):
        data = np.ascontiguousarray(data)
        num_samples = len(data) - seq_length
        shape = (num_samples, seq_length, data.shape[1])
        strides = (data.strides[0], data.strides[0], data.strides[1])
        X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        y = targets[seq_length:]
        if len(y) > len(X): y = y[:len(X)]
        if len(X) > len(y): X = X[:len(y)]
        return X, y

    X_train, y_train = create_sequences_vectorized(X_train_scaled, train_df['Target'].values, config.LOOKBACK_WINDOW)
    X_test, y_test = create_sequences_vectorized(X_test_scaled, test_df['Target'].values, config.LOOKBACK_WINDOW)

    return (torch.tensor(X_train.copy()).float(), torch.tensor(y_train.copy()).long(), 
            torch.tensor(X_test.copy()).float(), torch.tensor(y_test.copy()).long())

def entrenar():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n{'='*50}")
        print(f"🚀 MODO TURBO (ESTABLE): Usando Apple M4 (Metal)")
        print(f"{'='*50}\n")
    else:
        device = torch.device("cpu")
        print("⚠️ Usando CPU")
    
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    
    X_train, y_train, X_test, y_test = preparar_datos_sin_fuga(df)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    model = PhoenixLSTM(input_size=8, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    class_weights = torch.tensor([0.2, 2.0, 2.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"--- [TRAINING] Iniciando... ---")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- LIMPIEZA DE MEMORIA VRAM ---
        # Esto libera la memoria 'basura' que queda tras cada epoca
        if device.type == 'mps':
            torch.mps.empty_cache()
            gc.collect()
        # --------------------------------

        if (epoch+1) % 1 == 0: 
            elapsed = time.time() - start_time
            print(f"✅ Epoch {epoch+1}/{config.EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Tiempo: {elapsed:.1f}s")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"--- [EXITO] Guardado en {config.MODEL_SAVE_PATH} ---")

if __name__ == "__main__":
    entrenar()