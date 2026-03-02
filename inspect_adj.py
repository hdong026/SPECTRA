import pickle
import numpy as np
import scipy.sparse as sp
import os

# 尝试寻找文件
paths = [
    "datasets/PEMS04/adj_mx.pkl",
    "datasets/raw_data/PEMS04/adj_PEMS04.pkl"
]

file_path = None
for p in paths:
    if os.path.exists(p):
        file_path = p
        break

if file_path:
    print(f"Loading graph from {file_path}...")
    with open(file_path, "rb") as f:
        pickle_data = pickle.load(f)

    adj = None
    if isinstance(pickle_data, (list, tuple)): adj = pickle_data[2]
    elif isinstance(pickle_data, dict): adj = pickle_data['adj_mx']
    else: adj = pickle_data

    if sp.issparse(adj): adj = adj.toarray()
    adj = np.array(adj, dtype=np.float32)

    print(f"Stats: Min={adj.min():.4f}, Max={adj.max():.4f}, Mean={adj.mean():.4f}")
    
    if adj.max() > 10:
        print("\n[CRITICAL WARNING] This is a DISTANCE matrix! (Values > 1.0)")
        print("Using this directly for GFT/GCN creates an INVERTED GRAPH.")
        print("SOLUTION: Must apply Gaussian Kernel: W = exp(-dist^2 / sigma^2)")
    else:
        print("\n[OK] This looks like a SIMILARITY matrix (Values <= 1.0).")
else:
    print("Could not find adj_mx.pkl")