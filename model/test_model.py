import torch
import torch.nn as nn
import numpy as np
import joblib  # to load the saved scaler
import sys

# === Define model class ===
class RuntimePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Load the model ===
model = RuntimePredictor()
model.load_state_dict(torch.load("runtime_predictor.pt"))
model.eval()

# === Load the scaler ===
scaler = joblib.load("scaler.save")  # Must match training-time scaler

# === Define input (via command line or hardcoded for now) ===
# Format: threads, cpu_percent, memory_mb, num_threads, context_switches, uptime, algorithm_encoded
if len(sys.argv) == 7:
    input_data = [float(x) for x in sys.argv[1:]]
else:
    print("Usage: python test_model.py threads cpu_percent memory_mb num_threads context_switches uptime algorithm_encoded")
    print("Using default example...")
    input_data = [1, 99.5, 335.19, 2, 14, 6.7]  # ← change as needed

# === Scale and predict ===
input_scaled = scaler.transform([input_data])
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    predicted_runtime = model(input_tensor).item()

print(f"✅ Predicted Runtime: {predicted_runtime:.2f} seconds")

