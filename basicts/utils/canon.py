# basicts/utils/canon.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass
class CanonicalConfig:
    """Config for canonical subspace interface + reference calibration."""
    T: int                    # history length (e.g., 12)
    m: int                    # temporal basis dim (<= T)
    k: int                    # graph basis dim
    basis: str = "dct"        # currently support: "dct"
    eps: float = 1e-8         # numerical stability


# -----------------------------
# Time basis
# -----------------------------
def build_time_basis_dct(T: int, m: int, device: Optional[torch.device] = None,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II basis matrix B \in R^{T x m}.
    B[t,0] = sqrt(1/T)
    B[t,k] = sqrt(2/T) * cos(pi/T * (t+0.5) * k), for k>=1
    """
    assert 1 <= m <= T, f"m must be in [1, T], got m={m}, T={T}"
    dev = device if device is not None else torch.device("cpu")

    t = (torch.arange(T, device=dev, dtype=dtype) + 0.5).unsqueeze(1)  # [T,1]
    k = torch.arange(m, device=dev, dtype=dtype).unsqueeze(0)          # [1,m]
    B = torch.cos(torch.pi / T * t * k)                                 # [T,m]

    alpha = torch.full((m,), np.sqrt(2.0 / T), device=dev, dtype=dtype)
    alpha[0] = np.sqrt(1.0 / T)
    B = B * alpha.unsqueeze(0)
    return B


def build_time_basis(cfg: CanonicalConfig, device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if cfg.basis.lower() == "dct":
        return build_time_basis_dct(cfg.T, cfg.m, device=device, dtype=dtype)
    raise ValueError(f"Unknown time basis: {cfg.basis}")


# -----------------------------
# Graph basis (U_k)
# -----------------------------
def load_Uk(dataset_name: str, k: int, root: str = "datasets",
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load cached U_k from datasets/<dataset_name>/graph_state_Uk.npy
    Expected shape: [N, k] (or [N, >=k]).
    """
    path = os.path.join(root, dataset_name, "graph_state_Uk.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find U_k cache: {path}. "
            f"Please generate datasets/{dataset_name}/graph_state_Uk.npy first."
        )
    U = np.load(path)
    if U.ndim != 2:
        raise ValueError(f"U_k must be 2D array, got shape={U.shape}")
    if U.shape[1] < k:
        raise ValueError(f"U_k has k={U.shape[1]} < required k={k}")
    U = U[:, :k]
    t = torch.from_numpy(U).to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


# -----------------------------
# Canonical projection: Z = B^T X U
# -----------------------------
def project_to_Z(
    X_hist: torch.Tensor,
    Bm: torch.Tensor,
    Uk: torch.Tensor,
) -> torch.Tensor:
    """
    Project node-domain history to canonical Z.

    Args:
        X_hist: [B, T, N] or [T, N]
        Bm:     [T, m] (orthonormal time basis)
        Uk:     [N, k] (graph basis)

    Returns:
        Z: [B, m, k] or [m, k]
    """
    if X_hist.dim() == 2:
        # [T,N] -> [1,T,N]
        Xb = X_hist.unsqueeze(0)
        squeeze_back = True
    elif X_hist.dim() == 3:
        Xb = X_hist
        squeeze_back = False
    else:
        raise ValueError(f"X_hist must be 2D or 3D, got {X_hist.shape}")

    B, T, N = Xb.shape
    assert Bm.shape[0] == T, f"Bm T mismatch: {Bm.shape[0]} vs {T}"
    assert Uk.shape[0] == N, f"Uk N mismatch: {Uk.shape[0]} vs {N}"

    # Z[b,m,k] = sum_{t,n} Bm[t,m] * X[b,t,n] * Uk[n,k]
    Z = torch.einsum("tm,btn,nk->bmk", Bm, Xb, Uk)
    return Z[0] if squeeze_back else Z


# -----------------------------
# Energy spectrum M_G = E[Z^2]
# -----------------------------
@torch.no_grad()
def estimate_MG_from_dataloader(
    data_loader,
    Bm: torch.Tensor,
    Uk: torch.Tensor,
    flow_channel: int = 0,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Estimate M_G = E[ Z ⊙ Z ] for one dataset (graph).
    Uses *unlabeled* history slices from the given data_loader.

    Assumes each batch yields (future_data, history_data) where history_data is [B,T,N,C].
    Only uses history flow channel for X.

    Returns:
        M_G: [m, k] on CPU (float32)
    """
    Bm_dev = Bm.to(device) if device is not None else Bm
    Uk_dev = Uk.to(device) if device is not None else Uk

    sum_E = None
    count = 0

    for bi, batch in enumerate(data_loader):
        if max_batches is not None and bi >= max_batches:
            break

        # BasicTS dataset returns (future, history)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            _, history = batch[0], batch[1]
        else:
            raise ValueError("Unexpected batch format. Expect (future_data, history_data).")

        history = history.to(device) if device is not None else history
        # history: [B,T,N,C]
        if history.dim() != 4:
            raise ValueError(f"history_data must be 4D [B,T,N,C], got {history.shape}")

        X = history[..., flow_channel]  # [B,T,N]
        Z = project_to_Z(X, Bm_dev, Uk_dev)  # [B,m,k]
        E = Z * Z  # [B,m,k]

        batch_sum = E.sum(dim=0)  # [m,k]
        sum_E = batch_sum if sum_E is None else (sum_E + batch_sum)
        count += E.shape[0]

    if sum_E is None or count == 0:
        raise RuntimeError("No batches processed when estimating M_G.")

    M = sum_E / float(count)
    return M.detach().cpu().float()


def energy_dist_from_M(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert energy matrix M (>=0) to a probability distribution p (sum=1).
    """
    p = M.clamp(min=0.0)
    s = p.sum().clamp(min=eps)
    return p / s


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen–Shannon divergence for 2D distributions (flattened internally).
    Returns a scalar tensor.
    """
    p = p.reshape(-1).clamp(min=eps)
    q = q.reshape(-1).clamp(min=eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return 0.5 * (kl_pm + kl_qm)


def drift_js_from_M(M1: torch.Tensor, M2: torch.Tensor, eps: float = 1e-12) -> float:
    p = energy_dist_from_M(M1, eps=eps)
    q = energy_dist_from_M(M2, eps=eps)
    return float(js_divergence(p, q, eps=eps).detach().cpu())


# -----------------------------
# Reference calibration (Scheme B): Sinkhorn scaling
# -----------------------------
@torch.no_grad()
def sinkhorn_knopp(
    MG: torch.Tensor,
    r_t: torch.Tensor,
    r_g: torch.Tensor,
    iters: int = 50,
    eps: float = 1e-8,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve for positive scaling vectors a (m,) and b (k,) such that:
        diag(a) * MG * diag(b) has row sums r_t and col sums r_g.

    This is the classic Sinkhorn-Knopp matrix scaling iteration.

    Args:
        MG:  [m,k] nonnegative matrix (energy)
        r_t: [m]   target row marginals (positive)
        r_g: [k]   target col marginals (positive)

    Returns:
        a: [m] (float32)
        b: [k] (float32)
    """
    dev = device if device is not None else MG.device

    M = MG.to(dev).clamp(min=0.0)
    rt = r_t.to(dev).clamp(min=eps)
    rg = r_g.to(dev).clamp(min=eps)

    # Add a tiny floor to avoid zero-support issues (practical)
    M = M + eps

    b = torch.ones((M.shape[1],), device=dev, dtype=M.dtype)

    for _ in range(iters):
        Mb = torch.matmul(M, b).clamp(min=eps)         # [m]
        a = rt / Mb                                    # [m]
        MTa = torch.matmul(M.t(), a).clamp(min=eps)     # [k]
        b = rg / MTa                                   # [k]

    return a.detach().cpu().float(), b.detach().cpu().float()


def apply_calibration_Z(
    Z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Apply diagonal calibration: Z_tilde = diag(a) Z diag(b)

    Args:
        Z: [B,m,k] or [m,k]
        a: [m]
        b: [k]
    """
    if Z.dim() == 2:
        return Z * a.view(-1, 1) * b.view(1, -1)
    if Z.dim() == 3:
        return Z * a.view(1, -1, 1) * b.view(1, 1, -1)
    raise ValueError(f"Z must be 2D or 3D, got {Z.shape}")


# -----------------------------
# Cache helpers
# -----------------------------
def save_np(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in arrays.items()})


def load_np(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}