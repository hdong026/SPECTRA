import pickle
import numpy as np

file_path = "datasets/PEMS04/data_in12_out12.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)["processed_data"] # [L, N, 4]

print(f"Data Shape: {data.shape}")
channels = ["Flow (Norm)", "TOD", "DOW", "Prior"]

for i, name in enumerate(channels):
    c_data = data[..., i]
    print(f"Channel {i} [{name}]: Mean={np.mean(c_data):.4f}, Std={np.std(c_data):.4f}, Min={np.min(c_data):.4f}, Max={np.max(c_data):.4f}")

print("\nExpectation:")
print("- Flow: Mean~0, Std~1")
print("- Prior: Mean~0, Std~1 (If this is Mean~1.9, Std~3.4, GFT logic is broken)")