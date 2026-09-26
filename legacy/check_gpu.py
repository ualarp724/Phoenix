import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

if torch.backends.mps.is_available():
    print("✅ ¡ÉXITO! Tu Mac M4 está listo para aceleración.")
    print("El dispositivo se llama: 'mps'")
    try:
        x = torch.ones(1).to("mps")
        print("✅ Test de escritura en GPU: OK")
    except Exception as e:
        print(f"❌ Error escribiendo en GPU: {e}")
else:
    print("❌ ERROR: PyTorch no detecta tu GPU M4.")
    print("Estás usando la CPU (Lento).")
    print("Solución: Reinstala PyTorch con: pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu")