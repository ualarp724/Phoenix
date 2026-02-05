import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from phoenix_brain import PhoenixLSTM, preparar_secuencias
from phoenix_processor import PhoenixDataProcessor
import phoenix_config as config
import numpy as np

# 1. FUNCIÓN DE PÉRDIDA DE ALTA PRECISIÓN (Focal Loss)
# Esta función obliga a la IA a centrarse solo en ejemplos difíciles y seguros
class PrecisionLoss(nn.Module):
    def __init__(self, gamma=2):
        super(PrecisionLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def endurecer_cerebro():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- [CEO] Iniciando Endurecimiento de Precisión | {device} ---")

    # Cargar y procesar datos frescos de Vantage
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    X, y, _ = preparar_secuencias(df)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Cargar el modelo actual
    model = PhoenixLSTM(input_size=6, hidden_layers=config.HIDDEN_LAYERS, num_classes=3).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))

    # Optimizador ultra-lento para no romper lo que ya sabe, solo pulir
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    criterion = PrecisionLoss(gamma=3) # Gamma alto = Máxima exigencia de precisión

    model.train()
    for epoch in range(100): # 100 épocas de pulido intenso
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Hardening [{epoch+1}/100] - Loss: {total_loss/len(loader):.6f}")

    # Guardar el nuevo cerebro endurecido
    torch.save(model.to("cpu").state_dict(), "phoenix_brain_precision.pth")
    print("--- [EXITO] Cerebro de alta precisión generado ---")

if __name__ == "__main__":
    endurecer_cerebro()