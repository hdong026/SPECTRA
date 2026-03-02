# scripts/canon/calibrate_one.py
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from basicts.data.dataset import TimeSeriesForecastingDataset
from basicts.utils.canon import (
    CanonicalConfig, build_time_basis, load_Uk,
    estimate_MG_from_dataloader, save_np, load_np, sinkhorn_knopp
)

def build_train_loader(dataset_name: str, batch_size: int, num_workers: int, T: int, H: int):
    data_dir = os.path.join("datasets", dataset_name)
    data_file = os.path.join(data_dir, f"data_in{T}_out{H}.pkl")
    index_file = os.path.join(data_dir, f"index_in{T}_out{H}.pkl")
    ds = TimeSeriesForecastingDataset(data_file_path=data_file, index_file_path=index_file, mode="train")
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=False, drop_last=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--ref", type=str, default="datasets/canon_ref.npz")
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--basis", type=str, default="dct")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=400)
    ap.add_argument("--sinkhorn_iters", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        raise FileNotFoundError(f"reference file not found: {args.ref}")

    ref = load_np(args.ref)
    r_t = torch.from_numpy(ref["r_t"]).float()
    r_g = torch.from_numpy(ref["r_g"]).float()

    device = torch.device(args.device)
    cfg = CanonicalConfig(T=args.T, m=args.m, k=args.k, basis=args.basis)
    Bm = build_time_basis(cfg, device=device)
    Uk = load_Uk(args.dataset, k=args.k, root="datasets", device=device)

    loader = build_train_loader(args.dataset, args.batch_size, args.num_workers, args.T, args.H)
    MG = estimate_MG_from_dataloader(loader, Bm=Bm, Uk=Uk, max_batches=args.max_batches, device=device)

    out_M = os.path.join("datasets", args.dataset, "canon_M.npz")
    save_np(out_M, M=MG.numpy())
    print("[SAVE]", out_M)

    a, b = sinkhorn_knopp(MG=MG, r_t=r_t, r_g=r_g, iters=args.sinkhorn_iters, device=device)
    out_ab = os.path.join("datasets", args.dataset, "canon_ab.npz")
    save_np(out_ab, a=a.numpy(), b=b.numpy(), sinkhorn_iters=np.array([args.sinkhorn_iters]))
    print("[SAVE]", out_ab)
    print("Done.")

if __name__ == "__main__":
    main()