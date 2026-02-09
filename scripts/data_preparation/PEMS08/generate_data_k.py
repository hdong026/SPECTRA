import os
import sys
import shutil
import pickle
import argparse
import numpy as np

# 适配 BasicTS 路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from basicts.data.transform import standard_transform

def generate_data(args: argparse.Namespace):
    """
    带 Top-K 参数的数据生成函数
    """
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    output_dir = args.output_dir
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    steps_per_day = args.steps_per_day
    
    # �� 从参数获取 K 值
    TOP_K = args.top_k

    print(f"\n>>> Processing Data for K={TOP_K}...")
    print(f"    Output Dir: {output_dir}")

    # 1. Read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel] # [L, N, 1]

    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * args.train_ratio)
    valid_num_short = round(num_samples * args.valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short

    # 2. Generate Index
    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]

    # 3. Normalize Data (Standard Scaler)
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # -------------------------------------------------------------------------
    # �� New Logic: Calculate Prior with Configurable K
    # -------------------------------------------------------------------------
    print(f"    Generating Prior with FFT (Top-{TOP_K})...")
    
    train_end = train_index[-1][1]
    train_data = data[:train_end, :, 0]
    train_mean, train_std = np.mean(train_data), np.std(train_data)
    
    steps_per_week = steps_per_day * 7
    num_weeks = train_data.shape[0] // steps_per_week
    # 截断多余数据以整除周
    train_reshaped = train_data[:num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
    weekly_profile = np.mean(train_reshaped, axis=0) # [Steps_Week, N]
    
    # FFT Filtering
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    
    # 使用传入的 TOP_K 进行截断
    max_freq = len(fft_coeffs)
    real_k = min(TOP_K, max_freq)
    
    fft_coeffs[real_k:] = 0 # Low-pass
    
    smooth_weekly_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0)
    
    # Tile & Normalize
    prior_full = np.tile(smooth_weekly_profile, (l // steps_per_week + 1, 1))[:l]
    prior_norm = (prior_full - train_mean) / train_std
    prior_norm = prior_norm[..., np.newaxis]
    # -------------------------------------------------------------------------

    # 4. Concatenate Features
    feature_list = [data_norm]
    
    if args.tod:
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod_tiled = np.tile(np.array(tod), [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if args.dow:
        dow = [(i // steps_per_day) % 7 for i in range(data_norm.shape[0])]
        dow_tiled = np.tile(np.array(dow), [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
        
    feature_list.append(prior_norm) # Prior Channel

    processed_data = np.concatenate(feature_list, axis=-1)
    print(f"    Final Data Shape: {processed_data.shape}")

    # 5. Dump Data
    index = {"train": train_index, "valid": valid_index, "test": test_index}
    with open(output_dir + "/index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(index, f)

    data_dict = {"processed_data": processed_data}
    with open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(data_dict, f)
        
    # Copy Adj
    if os.path.exists(graph_file_path):
        shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        print("    Warning: Graph file not found, skipping adj copy.")

if __name__ == "__main__":
    # Settings (Defaults)
    # 请根据您的实际数据集名称修改这里，或者在命令行传参
    DATASET_NAME = "PEMS08" 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # 默认路径
    RAW_DATA_DIR = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME)
    
    parser = argparse.ArgumentParser()
    # 新增参数
    parser.add_argument("--top_k", type=int, default=32, help="Freq components K")
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    
    # 自动推断路径
    parser.add_argument("--data_file_path", type=str, default="")
    parser.add_argument("--graph_file_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    
    # 其它参数
    parser.add_argument("--history_seq_len", type=int, default=12)
    parser.add_argument("--future_seq_len", type=int, default=12)
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument("--tod", type=bool, default=True)
    parser.add_argument("--dow", type=bool, default=True)
    parser.add_argument("--target_channel", type=list, default=[0])
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    
    args = parser.parse_args()

    # 路径补全
    if not args.data_file_path:
        args.data_file_path = os.path.join(RAW_DATA_DIR, f"{args.dataset_name}.npz")
    if not args.graph_file_path:
        # PEMS08 可能是 adj_PEMS08.pkl 或者 PEMS08.csv，请检查您的 raw_data
        args.graph_file_path = os.path.join(RAW_DATA_DIR, f"adj_{args.dataset_name}.pkl")
        
    # 自动生成带后缀的输出目录: datasets/PEMS08_K32
    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "datasets", f"{args.dataset_name}_K{args.top_k}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    generate_data(args)