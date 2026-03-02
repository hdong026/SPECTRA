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

def load_adj(pkl_filename):
    """加载邻接矩阵并强制转换为对称归一化拉普拉斯矩阵"""
    try:
        with open(pkl_filename, 'rb') as f:
            pickle_data = pickle.load(f)
        
        adj = None
        if isinstance(pickle_data, (list, tuple)): adj = pickle_data[2]
        elif isinstance(pickle_data, dict): adj = pickle_data['adj_mx']
        else: adj = pickle_data
        
        if sp.issparse(adj): adj = adj.toarray()
        adj = np.array(adj, dtype=np.float32)
        
        # 1. 强制对称化 (Symmetrization) - 关键！
        adj = np.maximum(adj, adj.T)
        
        # 2. 归一化拉普拉斯 L = I - D^-1/2 A D^-1/2
        # 加自环
        adj = adj + np.eye(adj.shape[0])
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(adj.shape[0]) - d_mat_inv_sqrt @ sp.coo_matrix(adj) @ d_mat_inv_sqrt
        return L.toarray()
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None

def generate_data(args: argparse.Namespace):
    # 1. 读取原始数据
    data = np.load(args.data_file_path)["data"]
    data = data[..., args.target_channel] # [L, N, 1]
    l, n, f = data.shape
    
    # 2. 划分数据集
    num_samples = l - (args.history_seq_len + args.future_seq_len) + 1
    train_num = round(num_samples * args.train_ratio)
    
    index_list = []
    for t in range(args.history_seq_len, num_samples + args.history_seq_len):
        index_list.append((t-args.history_seq_len, t, t+args.future_seq_len))
    train_index = index_list[:train_num]
    valid_index = index_list[train_num: round(num_samples * (args.train_ratio + args.valid_ratio))]
    test_index = index_list[round(num_samples * (args.train_ratio + args.valid_ratio)):]

    # 3. 数据归一化 (Standard Scaler)
    # 这一步非常关键：我们要拿到归一化后的数据去算 Prior
    # 这样 Prior 天然就是 Normalized Scale，不需要后续再去乱缩放
    scaler = standard_transform
    data_norm = scaler(data, args.output_dir, train_index, args.history_seq_len, args.future_seq_len)

    # 在 data_norm 计算后
    print(f"DEBUG: Data Mean: {np.mean(data):.4f}, Max: {np.max(data):.4f}") # 检查原始数据是否是 Vehicles/Hour
    print(f"DEBUG: Norm Mean: {np.mean(data_norm):.4f}, Std: {np.std(data_norm):.4f}")
    # 如果 Norm Std 远大于 1，说明归一化失败
    
    print(f"Data Normalized. Mean: {np.mean(data_norm):.4f}, Std: {np.std(data_norm):.4f}")

    # =========================================================================
    # SOTA Prior Generation (Correct Math)
    # =========================================================================
    print("Generating Spatio-Temporal Spectral Prior...")
    
    # 3.1 提取周周期 (Weekly Profile)
    # 注意：我们直接用 data_norm (归一化后的数据) 来算！
    train_end_idx = train_index[-1][1]
    train_data_norm = data_norm[:train_end_idx, :, 0]
    
    steps_per_week = args.steps_per_day * 7
    num_weeks = train_data_norm.shape[0] // steps_per_week
    
    if num_weeks < 1:
        weekly_profile = np.tile(np.mean(train_data_norm, axis=0), (steps_per_week, 1))
    else:
        train_reshaped = train_data_norm[:num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
        weekly_profile = np.mean(train_reshaped, axis=0) # [2016, 307] (已经是 Normalized Scale)
    
    # 3.2 时域滤波 (FFT) - 去除高频噪声
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    top_k_time = int(len(fft_coeffs) * 0.10) # 保留前 10% 低频
    fft_coeffs[top_k_time:] = 0
    temporal_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0)
    
    # 3.3 空域滤波 (GFT) - 提取全局模式
    L = load_adj(args.graph_file_path)
    final_prior_profile = temporal_profile # Fallback

    if L is not None:
        try:
            print("Applying Graph Spectral Filtering...")
            vals, vecs = np.linalg.eigh(L)

            kmax = min(n, int(os.getenv("TOKEN_K_MAX", "256")))
            Uk = vecs[:, :kmax].astype(np.float32)
            eigvals = vals[:kmax].astype(np.float32)

            # sign canonicalization for reproducibility
            for i in range(Uk.shape[1]):
                j = np.argmax(np.abs(Uk[:, i]))
                if Uk[j, i] < 0:
                    Uk[:, i] *= -1.0

            np.save(os.path.join(args.output_dir, "graph_state_Uk.npy"), Uk)
            np.save(os.path.join(args.output_dir, "graph_state_eigvals.npy"), eigvals)
            
            # GFT: x_hat = U^T x
            # temporal_profile: [T, N] -> [N, T]
            gft_coeffs = np.dot(vecs.T, temporal_profile.T) 
            
            # Filter: 保留前 50% 的空间低频 (太少会丢失局部路况，太多会引入噪声)
            spatial_top_k = int(n * 0.5)
            gft_coeffs[spatial_top_k:, :] = 0
            
            # IGFT: x = U x_hat
            spatial_profile = np.dot(vecs, gft_coeffs).T # [T, N]
            
            final_prior_profile = spatial_profile
            print("GFT Done.")
        except Exception as e:
            print(f"GFT Failed: {e}")
    
    # 3.4 平铺到全长
    num_tiles = (l // steps_per_week) + 2
    prior_full = np.tile(final_prior_profile, (num_tiles, 1))[:l]
    
    # 3.5 最终 Prior
    # 关键：不需要再做 (x - mean)/std 了！
    # 因为我们本来就是用归一化数据算出来的，它已经在正确的 Scale 上了。
    prior_final = prior_full[..., np.newaxis]

    print(f"Prior Generated. Mean: {np.mean(prior_final):.4f}, Std: {np.std(prior_final):.4f}, Max: {np.max(prior_final):.4f}")
    
    # 检查 Scale 是否匹配
    if np.abs(np.mean(prior_final)) > 1.0 or np.std(prior_final) > 2.0:
         print("WARNING: Prior scale still looks weird. Please check adjacency matrix.")
    # =========================================================================

    # 4. 拼接特征
    feature_list = [data_norm]
    if args.tod:
        tod = np.array([i % args.steps_per_day / args.steps_per_day for i in range(l)])
        feature_list.append(np.tile(tod, [1, n, 1]).transpose((2, 1, 0)))
    if args.dow:
        dow = np.array([(i // args.steps_per_day) % 7 for i in range(l)])
        feature_list.append(np.tile(dow, [1, n, 1]).transpose((2, 1, 0)))
        
    feature_list.append(prior_final)
    
    processed_data = np.concatenate(feature_list, axis=-1) # [L, N, 4]
    
    # 5. 保存
    index = {"train": train_index, "valid": valid_index, "test": test_index}
    with open(args.output_dir + f"/index_in{args.history_seq_len}_out{args.future_seq_len}.pkl", "wb") as f:
        pickle.dump(index, f)
    with open(args.output_dir + f"/data_in{args.history_seq_len}_out{args.future_seq_len}.pkl", "wb") as f:
        pickle.dump({"processed_data": processed_data}, f)
    if os.path.exists(args.graph_file_path):
        shutil.copyfile(args.graph_file_path, args.output_dir + "/adj_mx.pkl")

if __name__ == "__main__":
    # 参数配置 (保持你的原样)
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288
    DATASET_NAME = "PEMS07"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    OUTPUT_DIR = os.path.join(project_root, "datasets", DATASET_NAME)
    DATA_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"{DATASET_NAME}.npz")
    GRAPH_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"adj_{DATASET_NAME}.pkl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--data_file_path", type=str, default=DATA_FILE_PATH)
    parser.add_argument("--graph_file_path", type=str, default=GRAPH_FILE_PATH)
    parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN)
    parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN)
    parser.add_argument("--steps_per_day", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--tod", type=bool, default=True)
    parser.add_argument("--dow", type=bool, default=True)
    parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO)
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir, exist_ok=True)
    generate_data(args)