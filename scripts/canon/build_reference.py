# scripts/canon/build_reference.py
from __future__ import annotations

import os
import sys
import argparse
from typing import Dict

import numpy as np
import torch

# ----------------------------------------------------------------------
# Ensure we import the *local repo* basicts, not the site-packages one.
# build_reference.py is at: <repo>/scripts/canon/build_reference.py
# so repo root is three levels up.
# ----------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import dataset from the actual module path (avoid relying on __init__.py exports)
from basicts.data.dataset import TimeSeriesForecastingDataset

from basicts.utils.canon import (
    CanonicalConfig,
    build_time_basis,
    load_Uk,
    estimate_MG_from_dataloader,
    save_np,
    sinkhorn_knopp,
    drift_js_from_M,
)

# -----------------------------
# Data loader helper
# -----------------------------
def build_train_loader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    input_len: int,
    output_len: int,
) -> torch.utils.data.DataLoader:
    data_dir = os.path.join("datasets", dataset_name)
    data_file = os.path.join(data_dir, f"data_in{input_len}_out{output_len}.pkl")
    index_file = os.path.join(data_dir, f"index_in{input_len}_out{output_len}.pkl")

    ds = TimeSeriesForecastingDataset(
        data_file_path=data_file,
        index_file_path=index_file,
        mode="train",
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="PEMS04,PEMS07,PEMS-BAY",
                        help="Comma-separated dataset names.")
    parser.add_argument("--k", type=int, default=64, help="Graph basis dim k (U_k).")
    parser.add_argument("--m", type=int, default=12, help="Time basis dim m (B_m).")
    parser.add_argument("--T", type=int, default=12, help="History length T.")
    parser.add_argument("--H", type=int, default=12, help="Forecast horizon H (for locating files).")
    parser.add_argument("--basis", type=str, default="dct", choices=["dct"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=300,
                        help="Max batches for estimating M_G. Increase for more accurate stats.")
    parser.add_argument("--sinkhorn_iters", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_root", type=str, default="datasets",
                        help="Where to save canon_ref.npz and per-dataset canon_*.npz.")
    parser.add_argument("--compute_drift", action="store_true", help="Also compute pairwise drift(JS).")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    assert len(datasets) >= 1

    device = torch.device(args.device)

    cfg = CanonicalConfig(T=args.T, m=args.m, k=args.k, basis=args.basis)
    Bm = build_time_basis(cfg, device=device)  # [T,m]

    # 1) Estimate M_G for each dataset
    M_dict: Dict[str, torch.Tensor] = {}
    for name in datasets:
        print(f"\n[1/3] Estimating M_G for {name} ...")
        Uk = load_Uk(name, k=args.k, device=device)  # [N,k]
        loader = build_train_loader(
            dataset_name=name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_len=args.T,
            output_len=args.H,
        )
        MG = estimate_MG_from_dataloader(
            data_loader=loader,
            Bm=Bm,
            Uk=Uk,
            flow_channel=0,
            max_batches=args.max_batches,
            device=device,
        )  # [m,k] on cpu
        M_dict[name] = MG

        out_path = os.path.join(args.out_root, name, "canon_M.npz")
        save_np(out_path, M=MG.numpy())
        print(f"  saved: {out_path}  shape={tuple(MG.shape)}")

    # 2) Compute global reference R and marginals
    print("\n[2/3] Building global reference R and marginals ...")
    Ms = torch.stack([M_dict[n] for n in datasets], dim=0)  # [G,m,k]
    R = Ms.mean(dim=0)                                      # [m,k]
    r_t = R.sum(dim=1)                                      # [m]
    r_g = R.sum(dim=0)                                      # [k]

    ref_path = os.path.join(args.out_root, "canon_ref.npz")
    save_np(
        ref_path,
        R=R.numpy(),
        r_t=r_t.numpy(),
        r_g=r_g.numpy(),
        meta=np.array([f"T={args.T}", f"m={args.m}", f"k={args.k}", f"basis={args.basis}"], dtype=object),
    )
    print(f"  saved: {ref_path}")

    # 3) Per-dataset Sinkhorn calibration vectors a,b (label-free)
    print("\n[3/3] Solving Sinkhorn calibration (a,b) for each dataset ...")
    for name in datasets:
        MG = M_dict[name]
        a, b = sinkhorn_knopp(
            MG=MG,
            r_t=r_t,
            r_g=r_g,
            iters=args.sinkhorn_iters,
            eps=1e-8,
            device=device,
        )
        ab_path = os.path.join(args.out_root, name, "canon_ab.npz")
        save_np(ab_path, a=a.numpy(), b=b.numpy(), sinkhorn_iters=np.array([args.sinkhorn_iters]))
        print(f"  saved: {ab_path}  a.shape={tuple(a.shape)} b.shape={tuple(b.shape)}")

    # Optional: compute pairwise drift(JS) matrix
    if args.compute_drift and len(datasets) >= 2:
        print("\n[Optional] Computing pairwise Drift(JS) matrix ...")
        drift = np.zeros((len(datasets), len(datasets)), dtype=np.float32)
        for i, di in enumerate(datasets):
            for j, dj in enumerate(datasets):
                drift[i, j] = drift_js_from_M(M_dict[di], M_dict[dj])
        print("Datasets:", datasets)
        print("Drift(JS):\n", drift)

        drift_path = os.path.join(args.out_root, "canon_drift_js.npy")
        np.save(drift_path, drift)
        print(f"  saved: {drift_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()