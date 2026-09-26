import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import time
import gc
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from phoenix_processor import PhoenixDataProcessor
from phoenix_metrics import TradingMetrics
import phoenix_config as config
from core.calibration import find_best_temperature, save_temperature
from core.mtf import add_mtf_features_multi
from phoenix_brain import preparar_secuencias_flat

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

# REPRODUCIBILIDAD
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# --- HARDWARE ---
if torch.backends.mps.is_available(): 
    DEVICE = torch.device("mps")
    print("🚀 GPU M4 ACTIVA")
else: 
    DEVICE = torch.device("cpu")
    print("⚠️  CPU MODE")

if config.CPU_THREADS:
    torch.set_num_threads(int(config.CPU_THREADS))
else:
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(int(cpu_count))

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

warnings.filterwarnings('ignore')

# --- EL CEREBRO (PHOENIX BRAIN) ---
class PhoenixLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layers=[256, 128], num_classes=3, dropout=0.3):
        super(PhoenixLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_layers[0], 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        self.bn = nn.BatchNorm1d(hidden_layers[0])
        self.fc = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[1], num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: 
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name: 
                param.data.fill_(0)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none", weight=alpha)

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def preparar_secuencias(df, test_size=0.3):
    """
    Prepara secuencias con SEPARACIÓN CORRECTA DE DATOS para evitar data leakage
    IMPORTANTE: El scaler se entrena SOLO con datos de training
    """
    features = config.FEATURES  # Usar features unificadas de config
    
    if 'Target' not in df.columns: 
        raise ValueError("Falta columna 'Target'")
    
    missing = [c for c in features if c not in df.columns]
    if missing: 
        raise KeyError(f"Faltan columnas: {missing}")
    
    data = df[features].values
    target = df['Target'].values
    
    # PASO 1: Dividir ANTES de escalar para evitar data leakage
    split_idx = int(len(data) * (1 - test_size))
    data_train = data[:split_idx]
    data_test = data[split_idx:]
    target_train = target[:split_idx]
    target_test = target[split_idx:]
    
    # PASO 2: Entrenar scaler SOLO con datos de entrenamiento
    scaler = StandardScaler()
    scaler.fit(data_train)
    
    # PASO 3: Aplicar scaler a ambos conjuntos
    data_train_scaled = scaler.transform(data_train)
    data_test_scaled = scaler.transform(data_test)
    
    # PASO 4: Crear secuencias
    def crear_secuencias(data_scaled, target_data):
        X, y = [], []
        lookback = config.LOOKBACK_WINDOW
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i-lookback:i])
            y.append(target_data[i])
        return X, y
    
    X_train, y_train = crear_secuencias(data_train_scaled, target_train)
    X_test, y_test = crear_secuencias(data_test_scaled, target_test)
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.long)
    
    return X_train, y_train, X_test, y_test, scaler

def calcular_pesos_clases(y_tensor):
    """Calcula pesos para balancear clases desbalanceadas"""
    classes, counts = np.unique(y_tensor.numpy(), return_counts=True)
    weights = 1.0 / counts
    weights_tensor = torch.tensor(weights / weights.sum(), dtype=torch.float32).to(DEVICE)
    
    if config.VERBOSE:
        print(f"⚖️  Pesos de Clases: {dict(zip(classes, weights_tensor.cpu().numpy()))}")
    
    return weights_tensor

# --- SIMULACIÓN Y ENTRENAMIENTO MEJORADO ---
def ejecutar_modo_dios():
    print(f"\n{'='*70}")
    print(f" [PHOENIX EVOLUTION] Entrenamiento con Validación Cruzada")
    print(f"{'='*70}")
    
    if config.MODEL_TYPE == "xgboost":
        return ejecutar_modo_xgboost()

    # 1. CARGAR Y PROCESAR DATOS
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = add_mtf_features_multi(processor.clean_and_prepare(), config.MTF_CONFIGS)
    
    if len(df) < config.LOOKBACK_WINDOW + 100:
        print(f"❌ Error: Insuficientes datos. Se necesitan {config.LOOKBACK_WINDOW + 100}")
        return
    
    print(f"\n📊 Datos cargados: {len(df)} velas")
    print(f"   Target distribution: {df['Target'].value_counts().to_dict()}")
    
    # 2. PREPARAR DATOS CON SEPARACIÓN CORRECTA
    print(f"\n🔧 Preparando secuencias...")
    X_train, y_train, X_test, y_test, scaler = preparar_secuencias(df, test_size=0.3)
    
    # GUARDAR SCALER (CRÍTICO)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)
    print(f"✅ Scaler guardado en {config.SCALER_SAVE_PATH}")
    
    # Crear loaders
    pin_memory = DEVICE.type == "cuda"
    loader_kwargs = {
        "num_workers": config.DATALOADER_WORKERS,
        "pin_memory": pin_memory,
        "persistent_workers": config.DATALOADER_WORKERS > 0,
    }
    if config.DATALOADER_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        **loader_kwargs,
    )
    
    def _state_dict_for_save(current_model):
        if hasattr(current_model, "_orig_mod"):
            return current_model._orig_mod.state_dict()
        return current_model.state_dict()

    def _load_state_dict_safe(current_model, state):
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        current_model.load_state_dict(state)

    # 3. CREAR MODELO
    print(f"\n🧠 Inicializando modelo...")
    model = PhoenixLSTM(
        input_size=config.INPUT_SIZE, 
        hidden_layers=config.HIDDEN_LAYERS, 
        num_classes=3,
        dropout=config.DROPOUT_RATE
    ).to(DEVICE)

    if config.USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    
    weights = calcular_pesos_clases(y_train)
    criterion = FocalLoss(alpha=weights, gamma=config.FOCAL_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    # 4. ENTRENAMIENTO CON EARLY STOPPING
    print(f"\n🚀 Entrenando modelo...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(config.EPOCHS):
        # Training
        total_loss = 0
        total_batches = len(train_loader)
        print(f"\n🧪 Época {epoch+1}/{config.EPOCHS} | Batches: {total_batches}")
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader, start=1):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx == 1 or batch_idx % max(1, total_batches // 5) == 0 or batch_idx == total_batches:
                pct = (batch_idx / total_batches) * 100
                avg_loss = total_loss / batch_idx
                print(f"   ↳ Batch {batch_idx}/{total_batches} ({pct:.0f}%) | Loss: {avg_loss:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                val_logits.append(logits.detach().cpu())
                val_labels.append(batch_y.detach().cpu())
                
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # Temperature scaling en validación
        best_temp = find_best_temperature(val_logits, val_labels)
        save_temperature(best_temp)
        val_losses.append(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(_state_dict_for_save(model), config.MODEL_BEST_SAVE_PATH)
            best_epoch = epoch
        else:
            patience_counter += 1
        
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Época {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}%")
        
        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n   ⏹️  Early Stopping en época {epoch+1} (paciencia agotada)")
            break
    
    # Cargar mejor modelo
    _load_state_dict_safe(model, torch.load(config.MODEL_BEST_SAVE_PATH))
    torch.save(_state_dict_for_save(model), config.MODEL_SAVE_PATH)
    print(f"\n✅ Modelo entrenado y guardado en {config.MODEL_SAVE_PATH}")
    print(f"   Mejor época: {best_epoch + 1} con Val Loss: {best_val_loss:.4f}")
    
def ejecutar_modo_xgboost():
    if xgb is None:
        print("❌ XGBoost no está instalado. Instala el paquete xgboost.")
        return

    print(f"\n{'='*70}")
    print(f" [PHOENIX EVOLUTION] Entrenamiento XGBoost")
    print(f"{'='*70}")

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = add_mtf_features_multi(processor.clean_and_prepare(), config.MTF_CONFIGS)

    if len(df) < config.LOOKBACK_WINDOW + 100:
        print(f"❌ Error: Insuficientes datos. Se necesitan {config.LOOKBACK_WINDOW + 100}")
        return

    print(f"\n📊 Datos cargados: {len(df)} velas")
    print(f"   Target distribution: {df['Target'].value_counts().to_dict()}")

    X_train, y_train, X_test, y_test, scaler = preparar_secuencias_flat(df, test_size=0.3)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)
    print(f"✅ Scaler guardado en {config.SCALER_SAVE_PATH}")

    model = xgb.XGBClassifier(**config.XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    joblib.dump(model, config.MODEL_SAVE_PATH)
    print(f"✅ Modelo XGBoost guardado en {config.MODEL_SAVE_PATH}")
    print(f"\n🏁 ENTRENAMIENTO XGBOOST COMPLETADO")
    return
    # 5. EVALUACIÓN FINAL EN TEST SET
    print(f"\n📈 Evaluación en Test Set...")
    model.eval()
    capital_history = [config.CAPITAL_INICIAL]
    capital = config.CAPITAL_INICIAL
    trades_log = []
    
    tensor_test = X_test.to(DEVICE)
    
    for i in range(len(tensor_test) - 20):
        if capital < config.CAPITAL_PROTECCIÓN:
            print(f"💀 Protección de capital activada")
            break
        
        window = tensor_test[i].unsqueeze(0)
        
        with torch.no_grad():
            logits = model(window)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            conf, pred = conf.item(), pred.item()
        
        if pred != 0 and conf > config.UMBRAL_CONFIANZA:
            # Simulación simplificada (usar phoenix_backtester para backtesting detallado)
            price = df.iloc[-(len(X_test) - i - config.LOOKBACK_WINDOW)]['Close']
            atr = df.iloc[-(len(X_test) - i - config.LOOKBACK_WINDOW)]['NATR'] * price / 100
            
            if atr < config.MIN_ATR_THRESHOLD:
                continue
            
            sl_dist = atr * config.ATR_SL_MULTIPLIER
            tp_dist = atr * config.ATR_TP_MULTIPLIER
            
            riesgo = capital * config.RIESGO_POR_OPERACION
            lotes = max(riesgo / (sl_dist * 100), 0.01)
            
            # Simulación rápida (mejor usar backtester)
            if pred == 1:
                tp_price = price + tp_dist
                sl_price = price - sl_dist
            else:
                tp_price = price - tp_dist
                sl_price = price + sl_dist
            
            # Asumimos que alcanza TP (para demo)
            pnl = (tp_dist * 100 * lotes) - (lotes * 0.06)  # Comisión
            capital += pnl
            capital_history.append(capital)
            trades_log.append({'price': price, 'pnl': pnl, 'direction': 'BUY' if pred == 1 else 'SELL'})
    
    # Métricas
    print(f"\n{'='*70}")
    print(f" RESULTADOS DEL ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"Capital Final: ${capital:.2f}")
    print(f"Retorno: {((capital - config.CAPITAL_INICIAL) / config.CAPITAL_INICIAL * 100):.2f}%")
    print(f"Total Operaciones: {len(trades_log)}")
    if len(trades_log) > 0:
        metrics = TradingMetrics(config.CAPITAL_INICIAL, trades_log)
        metrics.imprimir_reporte(np.array(capital_history), "SIMULACIÓN EN TEST SET")
    print(f"{'='*70}\n")
    
    # Cleanup
    del model, optimizer, criterion, train_loader, val_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__ == "__main__":
    ejecutar_modo_dios()