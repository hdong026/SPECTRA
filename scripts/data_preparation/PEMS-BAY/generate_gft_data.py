import os
import sys
import shutil
import pickle
import argparse
import scipy.sparse as sp
import numpy as np

# ===== NEW: for .h5 reading =====
import pandas as pd
import h5py

# 适配 BasicTS 路径
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from basicts.data.transform import standard_transform


# ============================================================
# Utils
# ============================================================
def load_adj_as_laplacian(pkl_filename):
    """Load adjacency from pkl (py2/py3 compatible) and return normalized Laplacian."""
    try:
        with open(pkl_filename, "rb") as f:
            try:
                pickle_data = pickle.load(f)
            except UnicodeDecodeError:
                # python2 pickle in python3
                f.seek(0)
                pickle_data = pickle.load(f, encoding="latin1")

        adj = None
        if isinstance(pickle_data, (list, tuple)):
            adj = pickle_data[2]
        elif isinstance(pickle_data, dict):
            adj = pickle_data.get("adj_mx", None)
        else:
            adj = pickle_data

        if adj is None:
            raise ValueError("adj not found in pkl")

        if sp.issparse(adj):
            adj = adj.toarray()
        adj = np.array(adj, dtype=np.float32)

        # 1) symmetrize
        adj = np.maximum(adj, adj.T)

        # 2) normalized Laplacian: L = I - D^-1/2 (A+I) D^-1/2
        adj = adj + np.eye(adj.shape[0], dtype=np.float32)
        d = np.array(adj.sum(1), dtype=np.float32)
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        L = sp.eye(adj.shape[0], dtype=np.float32) - d_mat_inv_sqrt @ sp.coo_matrix(adj) @ d_mat_inv_sqrt
        return L.toarray().astype(np.float32)

    except Exception as e:
        print(f"[ERROR] load_adj_as_laplacian failed: {e}")
        return None


def canonicalize_sign(U):
    """Fix sign ambiguity for eigenvectors for reproducibility."""
    U = U.copy()
    for i in range(U.shape[1]):
        j = np.argmax(np.abs(U[:, i]))
        if U[j, i] < 0:
            U[:, i] *= -1.0
    return U


def load_pems_bay_raw(data_file_path: str) -> np.ndarray:
    """
    Load PEMS-BAY raw data from .h5 (commonly a pandas DataFrame).
    Returns: data [L, N, 1] float32
    """
    suffix = os.path.splitext(data_file_path)[1].lower()
    if suffix in [".npz"]:
        obj = np.load(data_file_path)
        if "data" in obj:
            data = obj["data"]
        else:
            # fallback: take the first array
            key = list(obj.keys())[0]
            data = obj[key]
        data = np.array(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[..., None]
        return data

    if suffix in [".h5", ".hdf5", ".hdf"]:
        # Try pandas first (most common in PEMS-BAY)
        try:
            df = pd.read_hdf(data_file_path)
            # df shape: [L, N]
            values = df.values.astype(np.float32)
            data = values[..., None]  # [L,N,1]
            return data
        except Exception as e_pd:
            print(f"[WARN] pd.read_hdf failed: {e_pd}. Try h5py...")

        # Fallback to h5py: grab the first dataset-like array
        with h5py.File(data_file_path, "r") as f:
            # find a dataset
            def _find_dataset(g):
                for k in g.keys():
                    if isinstance(g[k], h5py.Dataset):
                        return g[k][...]
                    if isinstance(g[k], h5py.Group):
                        out = _find_dataset(g[k])
                        if out is not None:
                            return out
                return None

            arr = _find_dataset(f)
            if arr is None:
                raise RuntimeError("Cannot find dataset inside h5 file.")
            arr = np.array(arr, dtype=np.float32)
            # common: [L,N] or [L,N,1]
            if arr.ndim == 2:
                arr = arr[..., None]
            return arr

    raise ValueError(f"Unsupported raw data format: {data_file_path}")


# ============================================================
# Main pipeline (match PEMS04 style)
# ============================================================
def generate_data(args: argparse.Namespace):
    # 1) Load raw
    raw = load_pems_bay_raw(args.data_file_path)  # [L,N,1] usually
    # Support selecting channels if raw has multiple channels
    if raw.ndim == 3 and raw.shape[-1] > 1:
        raw = raw[..., args.target_channel]  # list like [0]
    elif raw.ndim == 3 and raw.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"Unexpected raw shape: {raw.shape}")

    raw = raw.astype(np.float32)
    l, n, f = raw.shape
    print(f"[INFO] Raw loaded: shape={raw.shape} (L={l}, N={n}, C={f})")

    # 2) Split indices
    num_samples = l - (args.history_seq_len + args.future_seq_len) + 1
    train_num = round(num_samples * args.train_ratio)

    index_list = []
    for t in range(args.history_seq_len, num_samples + args.history_seq_len):
        index_list.append((t - args.history_seq_len, t, t + args.future_seq_len))

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: round(num_samples * (args.train_ratio + args.valid_ratio))]
    test_index = index_list[round(num_samples * (args.train_ratio + args.valid_ratio)) :]

    # 3) Normalize (use normalized data to build prior, same as your PEMS04)
    scaler = standard_transform
    data_norm = scaler(raw, args.output_dir, train_index, args.history_seq_len, args.future_seq_len)

    print(f"DEBUG: Raw Mean: {np.mean(raw):.4f}, Max: {np.max(raw):.4f}")
    print(f"DEBUG: Norm Mean: {np.mean(data_norm):.4f}, Std: {np.std(data_norm):.4f}")
    print(f"Data Normalized. Mean: {np.mean(data_norm):.4f}, Std: {np.std(data_norm):.4f}")

    # =========================================================================
    # SOTA Prior Generation (same math as PEMS04)
    # =========================================================================
    print("Generating Spatio-Temporal Spectral Prior...")

    # 3.1 weekly profile from normalized TRAIN region
    train_end_idx = train_index[-1][1]
    train_data_norm = data_norm[:train_end_idx, :, 0]  # [Ttrain, N]

    steps_per_week = args.steps_per_day * 7
    num_weeks = train_data_norm.shape[0] // steps_per_week

    if num_weeks < 1:
        weekly_profile = np.tile(np.mean(train_data_norm, axis=0), (steps_per_week, 1)).astype(np.float32)
    else:
        train_reshaped = train_data_norm[: num_weeks * steps_per_week].reshape(num_weeks, steps_per_week, n)
        weekly_profile = np.mean(train_reshaped, axis=0).astype(np.float32)  # [steps_per_week, N]

    # 3.2 temporal FFT low-pass
    fft_coeffs = np.fft.rfft(weekly_profile, axis=0)
    top_k_time = int(len(fft_coeffs) * 0.10)  # keep low 10%
    fft_coeffs[top_k_time:] = 0
    temporal_profile = np.fft.irfft(fft_coeffs, n=weekly_profile.shape[0], axis=0).astype(np.float32)  # [Tweek,N]

    # 3.3 spatial GFT low-pass (via Laplacian eigenvectors)
    L = load_adj_as_laplacian(args.graph_file_path)
    final_prior_profile = temporal_profile  # fallback

    if L is not None:
        try:
            print("Applying Graph Spectral Filtering...")
            vals, vecs = np.linalg.eigh(L)  # L symmetric
            # sort by eigenvalues ascending
            idx = np.argsort(vals)
            vals = vals[idx].astype(np.float32)
            vecs = vecs[:, idx].astype(np.float32)

            # sign canonicalization for reproducibility
            vecs = canonicalize_sign(vecs)

            # GFT: x_hat = U^T x  (temporal_profile: [T,N])
            gft_coeffs = np.dot(vecs.T, temporal_profile.T)  # [N,T]

            spatial_top_k = int(n * 0.5)  # keep low 50%
            gft_coeffs[spatial_top_k:, :] = 0

            spatial_profile = np.dot(vecs, gft_coeffs).T.astype(np.float32)  # [T,N]
            final_prior_profile = spatial_profile
            print("GFT Done.")

            # ---- [OPTIONAL but recommended] materialize graph state for SPECTRA-GFM ----
            kmax = min(n, int(os.getenv("TOKEN_K_MAX", "256")))
            Uk = vecs[:, :kmax].astype(np.float32)
            eigvals = vals[:kmax].astype(np.float32)
            np.save(os.path.join(args.output_dir, "graph_state_Uk.npy"), Uk)
            np.save(os.path.join(args.output_dir, "graph_state_eigvals.npy"), eigvals)
            print(f"[GRAPH_STATE] Saved Uk/eigvals: Uk={Uk.shape}, eigvals={eigvals.shape}")

        except Exception as e:
            print(f"GFT Failed: {e}")

    # 3.4 tile to full length
    num_tiles = (l // steps_per_week) + 2
    prior_full = np.tile(final_prior_profile, (num_tiles, 1))[:l].astype(np.float32)

    # 3.5 prior_final: already in normalized scale
    prior_final = prior_full[..., np.newaxis]  # [L,N,1]

    print(
        f"Prior Generated. Mean: {np.mean(prior_final):.4f}, Std: {np.std(prior_final):.4f}, Max: {np.max(prior_final):.4f}"
    )
    if np.abs(np.mean(prior_final)) > 1.0 or np.std(prior_final) > 2.0:
        print("WARNING: Prior scale still looks weird. Please check adjacency matrix.")

    # =========================================================================

    # 4) concatenate features: [flow_norm, tod, dow, prior]
    feature_list = [data_norm]

    if args.tod:
        tod = np.array([i % args.steps_per_day / args.steps_per_day for i in range(l)], dtype=np.float32)
        feature_list.append(np.tile(tod, [1, n, 1]).transpose((2, 1, 0)))

    if args.dow:
        dow = np.array([(i // args.steps_per_day) % 7 for i in range(l)], dtype=np.float32)
        feature_list.append(np.tile(dow, [1, n, 1]).transpose((2, 1, 0)))

    feature_list.append(prior_final)

    processed_data = np.concatenate(feature_list, axis=-1).astype(np.float32)  # [L,N,4]

    # 5) save
    index = {"train": train_index, "valid": valid_index, "test": test_index}

    with open(os.path.join(args.output_dir, f"index_in{args.history_seq_len}_out{args.future_seq_len}.pkl"), "wb") as f:
        pickle.dump(index, f)

    with open(os.path.join(args.output_dir, f"data_in{args.history_seq_len}_out{args.future_seq_len}.pkl"), "wb") as f:
        pickle.dump({"processed_data": processed_data}, f)

    # keep adjacency for model
    if os.path.exists(args.graph_file_path):
        shutil.copyfile(args.graph_file_path, os.path.join(args.output_dir, "adj_mx.pkl"))

    print(f"[DONE] Saved to {args.output_dir}")
    print(f"  processed_data: {processed_data.shape}")
    print(f"  index sizes: train={len(train_index)} valid={len(valid_index)} test={len(test_index)}")


if __name__ == "__main__":
    # 参数配置（保持你原样）
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]
    STEPS_PER_DAY = 288
    DATASET_NAME = "PEMS-BAY"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    OUTPUT_DIR = os.path.join(project_root, "datasets", DATASET_NAME)
    DATA_FILE_PATH = os.path.join(project_root, "datasets", "raw_data", DATASET_NAME, f"{DATASET_NAME}.h5")
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
    os.makedirs(args.output_dir, exist_ok=True)
    generate_data(args)