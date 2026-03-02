import os
import sys
import shutil
import pickle
import argparse
import scipy.sparse as sp
import numpy as np

# 适配 BasicTS 路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from basicts.data.transform import standard_transform

def check_stats(name, arr):
    """数值探针：打印统计信息"""
    print(f"[{name}] Shape: {arr.shape} | Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f}")

def load_adj(pkl_filename):
    """鲁棒的邻接矩阵加载"""
    try:
        with open(pkl_filename, 'rb') as f:
            pickle_data = pickle.load(f)
        
        adj = None
        if isinstance(pickle_data, (list, tuple)) and len(pickle_data) == 3:
            adj = pickle_data[2]
        elif isinstance(pickle_data, dict) and 'adj_mx' in pickle_data:
            adj = pickle_data['adj_mx']
        elif isinstance(pickle_data, (np.ndarray, sp.spmatrix)):
            adj = pickle_data
        
        if adj is not None:
            if sp.issparse(adj): adj = adj.toarray()
            adj = np.array(adj, dtype=np.float32)
            check_stats("Loaded Adjacency Matrix", adj)
            return adj
        return None
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def calculate_normalized_laplacian(adj):
    """计算拉普拉斯矩阵，带自环和对称化检查"""
    # 强制对称化 (对于有向图 PEMS04 必须做，否则特征值会有虚数)
    adj = np.maximum(adj, adj.T)
    
    # 归一化逻辑 L = I - D^-1/2 A D^-1/2
    adj = sp.coo_matrix(adj + np.eye(adj.shape[0])) # 加自环
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def generate_data(args: argparse.Namespace):
    # 1. 读取数据
    data = np.load(args.data_file_path)["data"]
    data = data[..., args.target_channel]
    
    # 2. 划分数据集
    l, n, f = data.shape
    num_samples = l - (args.history_seq_len + args.future_seq_len) + 1
    train_num = round(num_samples * args.train_ratio)
    
    # 生成索引
    index_list = []
    for t in range(args.history_seq_len, num_samples + args.history_seq_len):
        index_list.append((t-args.history_seq_len, t, t+args.future_seq_len))
    train_index = index_list[:train_num]

    # 3. 数据归一化
    # 这里的 scaler 会计算训练集的 mean/std
    print("\n>>> 1. Processing Raw Data...")
    scaler = standard_transform
    data_norm = scaler(data, args.output_dir, train_index, args.history_seq_len, args.future_seq_len)
    check_stats("Normalized Train Data", data_norm)

    # 获取训练集统计量用于后续对比
    train_end = train_index[-1][1]
    train_data_raw = data[:train_end, :, 0]
    train_mean = np.mean(train_data_raw)
    train_std = np.std(train_data_raw)
    print(f"   (Raw Train Mean: {train_mean:.4f}, Raw Train Std: {train_std:.4f})")

    # =========================================================================
    # DEBUG: Spatio-Temporal Spectral Filtering
    # =========================================================================
    print("\n>>> 2. Generating Prior (Debugging Mode)...")
    
    # 3.1 Weekly Profile
    steps_per_week = args.steps_per_day * 7
    num_weeks = train_data_raw.shape[0] // steps_per_week
    train_reshaped = train_data_raw[:num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
    weekly_profile = np.mean(train_reshaped, axis=0)
    check_stats("Weekly Profile (Raw Scale)", weekly_profile)

    # 3.2 Temporal FFT
    print("   -> Applying FFT...")
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    # Filter top 5%
    top_k_time = int(len(fft_coeffs) * 0.05)
    fft_coeffs[top_k_time:] = 0
    temporal_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0)
    check_stats("Temporal Smooth Profile", temporal_profile)

    # 3.3 Spatial GFT (The Suspect)
    print("   -> Applying GFT (The Suspect)...")
    adj_mx = load_adj(args.graph_file_path)
    
    final_prior = temporal_profile # Default fallback

    if adj_mx is not None:
        try:
            L = calculate_normalized_laplacian(adj_mx)
            vals, vecs = np.linalg.eigh(L.toarray())
            print(f"      Eigenvalues range: [{vals.min():.4f}, {vals.max():.4f}]")
            
            # check orthogonality
            ortho_check = np.dot(vecs.T, vecs)
            is_ortho = np.allclose(ortho_check, np.eye(n), atol=1e-4)
            print(f"      Eigenvectors Orthogonality: {is_ortho}")

            # GFT Projection: U^T x
            # temporal_profile: [T, N] -> [N, T]
            gft_coeffs = np.dot(vecs.T, temporal_profile.T) 
            check_stats("GFT Coefficients (Spectral Domain)", gft_coeffs)

            # Filter: Keep low freq spatial
            spatial_top_k = int(n * 0.3)
            gft_coeffs[spatial_top_k:, :] = 0
            
            # IGFT Reconstruction: U x_hat
            spatial_profile = np.dot(vecs, gft_coeffs).T
            check_stats("Spatial Smooth Profile (Reconstructed)", spatial_profile)
            
            final_prior = spatial_profile
            
        except Exception as e:
            print(f"!!! GFT Failed: {e}")

    # 3.4 Normalization Check
    print("\n>>> 3. Checking Prior Normalization...")
    # 模拟生成的 Prior
    num_tiles = (l // steps_per_week) + 1
    prior_full = np.tile(final_prior, (num_tiles, 1))[:l]
    
    # 关键检查点：使用和训练数据相同的 Mean/Std 进行归一化
    prior_norm = (prior_full - train_mean) / (train_std + 1e-5)
    check_stats("Final Normalized Prior Channel", prior_norm)
    
    # 对比检查
    diff_mean = np.abs(np.mean(prior_norm) - np.mean(data_norm))
    print(f"\n[DIAGNOSIS RESULT]:")
    print(f"Data Mean: {np.mean(data_norm):.4f} vs Prior Mean: {np.mean(prior_norm):.4f}")
    if np.abs(np.mean(prior_norm)) > 10 or np.std(prior_norm) > 10:
        print("�� CRITICAL WARNING: Prior scale is EXPLODED! GFT or Normalization is wrong.")
    elif diff_mean > 1.0:
        print("�� WARNING: Prior distribution shifted significantly from Data.")
    else:
        print("�� PASSED: Prior scale matches Data scale. Innovation preserved!")

if __name__ == "__main__":
    # 配置参数 (直接写死测试)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        output_dir="./debug_output",
        data_file_path="/home/dhz/KASA-ST/datasets/raw_data/PEMS04/PEMS04.npz", # 请确认路径
        graph_file_path="/home/dhz/KASA-ST/datasets/raw_data/PEMS04/adj_PEMS04.pkl", # 请确认路径
        history_seq_len=12, future_seq_len=12,
        steps_per_day=288, tod=True, dow=True,
        target_channel=[0], train_ratio=0.7, valid_ratio=0.1
    )
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    generate_data(args)