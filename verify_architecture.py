import torch, sys
sys.path.insert(0, r'C:\Users\user\Desktop\PumpSmart_Project\src')
from model_architecture import LSTMAutoencoder

m = LSTMAutoencoder()
state = torch.load(
    r"C:\Users\user\Desktop\PumpSmart_Project\models\lstm_ae_baseline_best.pth",
    map_location='cpu', weights_only=True
)
m.load_state_dict(state, strict=True)
m.eval()

with torch.no_grad():
    normal_in = torch.ones(1, 50, 8)
    fault_in  = torch.full((1, 50, 8), 4.0)
    mae_n = torch.abs(m(normal_in) - normal_in).mean().item()
    mae_f = torch.abs(m(fault_in)  - fault_in ).mean().item()

print(f"✅ Load: strict=True PASSED")
print(f"MAE on normal (all 1.0): {mae_n:.4f}  → expect < 0.11")
print(f"MAE on fault  (all 4.0): {mae_f:.4f}  → expect > 0.11")
print("Gate 3 ACTIVE ✅" if mae_f > 0.11 and mae_n < 0.11 else "❌ Something wrong — paste output")