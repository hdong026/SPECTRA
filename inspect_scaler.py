import pickle
import os
import numpy as np

# 替换成你日志中显示的路径
file_path = "datasets/PEMS04/scaler_in12_out12.pkl"

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list)):
                # 如果是数组，打印形状和部分值
                v_arr = np.array(v)
                print(f"Key '{k}': Shape={v_arr.shape}, Mean={np.mean(v_arr):.4f}")
            else:
                print(f"Key '{k}': {v}")
    else:
        print("Data content:", data)
else:
    print(f"File not found: {file_path}")