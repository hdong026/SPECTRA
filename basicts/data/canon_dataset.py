# basicts/data/canon_dataset.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils import load_pkl
from ..utils.canon import CanonicalConfig, build_time_basis, load_Uk, project_to_Z, apply_calibration_Z, load_np


class CanonicalHistoryDataset(Dataset):
    """
    Dataset that:
      - reads node-domain processed flow data (normalized) from BasicTS pkl
      - converts history flow [T,N] -> canonical Z [m,k] using Z = B^T X U_k
      - applies label-free reference calibration: Z_tilde = diag(a) Z diag(b)
      - returns:
          future_data: [H,N,1] (node domain, for loss/metrics)
          history_data: [m,k,1] (canonical domain, fixed-size input)
    """

    def __init__(
        self,
        data_file_path: str,
        index_file_path: str,
        mode: str,
        dataset_name: Optional[str] = None,
        # canonical dims (defaults match your typical T=12,H=12,k=64,m=12)
        T: int = 12,
        H: int = 12,
        m: int = 12,
        k: int = 64,
        basis: str = "dct",
        flow_channel: int = 0,
        cache_root: str = "datasets",
        use_calib: Optional[bool] = None,
    ) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)

        # infer dataset_name from path if not provided: datasets/<NAME>/data_in...
        if dataset_name is None:
            dataset_name = os.path.basename(os.path.dirname(os.path.abspath(data_file_path)))
        self.dataset_name = dataset_name

        self.T = int(T)
        self.H = int(H)
        self.m = int(m)
        self.k = int(k)
        self.basis = str(basis)
        self.flow_channel = int(flow_channel)
        self.cache_root = cache_root

        # env switch (default: enabled)
        if use_calib is None:
            use_calib = os.getenv("USE_CALIB", "1").lower() in ["1", "true", "yes", "y"]
        self.use_calib = bool(use_calib)

        # ---- load processed data ----
        # processed_data: [L, N, C]
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        # keep only flow channel (normalized), shape [L,N,1]
        flow = processed_data[..., self.flow_channel:self.flow_channel + 1]
        self.data = torch.from_numpy(flow).float()  # [L,N,1]

        # ---- load indices ----
        self.index = load_pkl(index_file_path)[mode]

        # ---- few-shot support (same policy as your original dataset) ----
        env_mode = os.getenv("MODE", "train").lower()
        fewshot_ratio = float(os.getenv("FEWSHOT_RATIO", "1.0"))
        if mode == "train" and env_mode == "finetune" and fewshot_ratio < 1.0:
            full_len = len(self.index)
            keep = max(1, int(full_len * fewshot_ratio))
            self.index = self.index[:keep]

        # ---- precompute B_m and load U_k on CPU ----
        cfg = CanonicalConfig(T=self.T, m=self.m, k=self.k, basis=self.basis)
        self.Bm = build_time_basis(cfg, device=torch.device("cpu"))  # [T,m]
        self.Uk = load_Uk(self.dataset_name, k=self.k, root=self.cache_root,
                          device=torch.device("cpu"))                # [N,k]

        # ---- load cached calibration (a,b) if available ----
        self.a, self.b = self._load_ab_or_default()

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"BasicTS can not find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"BasicTS can not find index file {index_file_path}")

    def _load_ab_or_default(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a,b from datasets/<name>/canon_ab.npz if exists.
        Otherwise return ones.
        """
        a = torch.ones((self.m,), dtype=torch.float32)
        b = torch.ones((self.k,), dtype=torch.float32)

        if not self.use_calib:
            return a, b

        ab_path = os.path.join(self.cache_root, self.dataset_name, "canon_ab.npz")
        if not os.path.exists(ab_path):
            # no calibration cached yet, fallback silently (but still works)
            return a, b

        obj = load_np(ab_path)
        if "a" in obj and "b" in obj:
            a_np = obj["a"].astype(np.float32).reshape(-1)
            b_np = obj["b"].astype(np.float32).reshape(-1)
            if a_np.shape[0] == self.m and b_np.shape[0] == self.k:
                a = torch.from_numpy(a_np)
                b = torch.from_numpy(b_np)
        return a, b

    def __getitem__(self, index: int) -> tuple:
        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # history: [T,N,1], future: [H,N,1]
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            history_index = list(idx[0])  # copy
            assert idx[1] not in history_index, "current time t should not included in idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]

        # history_data: [T,N,1] -> X: [T,N]
        X = history_data[..., 0]  # [T,N]

        # canonical projection: Z [m,k]
        Z = project_to_Z(X_hist=X, Bm=self.Bm, Uk=self.Uk)  # [m,k] (cpu)

        # apply calibration (label-free, cached)
        if self.use_calib:
            Z = apply_calibration_Z(Z, self.a, self.b)

        # model expects 4D tensor [L, N, C] style -> here [m,k,1]
        history_Z = Z.unsqueeze(-1)  # [m,k,1]

        return future_data, history_Z

    def __len__(self):
        return len(self.index)