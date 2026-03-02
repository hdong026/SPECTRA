# basicts/archs/arch_zoo/KASA_arch_v2/KASA_CANON_GFM.py
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

# Reuse your existing MLP block (already in kasa_components_tkde.py)
from basicts.archs.arch_zoo.KASA_arch_v2.kasa_components_tkde import MultiLayerPerceptron


def _lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


class PatchEncoderZ(nn.Module):
    """
    Patch encoder in canonical Z-space.
    Input:  patch_input [B, P, L, K, 1] where L=patch_len, K=k
    Output: pred_k      [B, H, K, 1]
    """

    def __init__(self, patch_len: int, num_patches: int, d: int, output_len: int, num_layer: int):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches
        self.output_len = output_len

        # embed each patch (channels = patch_len)
        self.data_embedding = nn.Conv2d(in_channels=patch_len, out_channels=d, kernel_size=(1, 1), bias=True)
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(d, d) for _ in range(num_layer)])

        # project flattened patches -> horizon channels
        self.proj = nn.Conv2d(in_channels=d * num_patches, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, patch_input: torch.Tensor) -> torch.Tensor:
        # patch_input: [B,P,L,K,1]
        x = patch_input.squeeze(-1)                      # [B,P,L,K]
        x = x.permute(0, 2, 1, 3).contiguous()          # [B,L,P,K] (L as channels)
        x = self.data_embedding(x)                      # [B,d,P,K]
        x = self.encoder(x)                             # [B,d,P,K]
        x = x.reshape(x.shape[0], -1, x.shape[3], 1)    # [B,d*P,K,1]
        pred = self.proj(x)                             # [B,H,K,1]
        return pred


class DownsampEncoderZ(nn.Module):
    """
    Downsample encoder in canonical Z-space.
    Input:  ds_input [B, S, Ld, K, 1] where S=stride, Ld = (m_pad / stride)
    Output: pred_k   [B, H, K, 1]
    """

    def __init__(self, ds_len: int, stride: int, d: int, output_len: int, num_layer: int):
        super().__init__()
        self.ds_len = ds_len
        self.stride = stride
        self.output_len = output_len

        # treat ds_len as channels
        self.data_embedding = nn.Conv2d(in_channels=ds_len, out_channels=d, kernel_size=(1, 1), bias=True)
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(d, d) for _ in range(num_layer)])

        # project flattened S segments -> horizon channels
        self.proj = nn.Conv2d(in_channels=d * stride, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, ds_input: torch.Tensor) -> torch.Tensor:
        # ds_input: [B,S,Ld,K,1]
        x = ds_input.squeeze(-1)                         # [B,S,Ld,K]
        x = x.permute(0, 2, 1, 3).contiguous()          # [B,Ld,S,K]
        x = self.data_embedding(x)                      # [B,d,S,K]
        x = self.encoder(x)                             # [B,d,S,K]
        x = x.reshape(x.shape[0], -1, x.shape[3], 1)    # [B,d*S,K,1]
        pred = self.proj(x)                             # [B,H,K,1]
        return pred


class KASA_CANON_GFM(nn.Module):
    """
    Canonical ST-GFM backbone:
      - input is already canonical Z-space: [B, m, k, 1]
      - predict in k-mode space: [B, H, k, 1]
      - reconstruct to node domain using U_k (attached per dataset): [B, H, N, 1]

    NOTE:
      - U_k is graph-specific, injected by runner via set_graph_state(Uk).
      - We store U_k as a non-persistent buffer to avoid checkpoint shape coupling.
    """

    def __init__(self, **model_args):
        super().__init__()

        # canonical "sequence length" (m) and "mode tokens" (k)
        self.input_len = int(model_args.get("input_len", 12))     # this is m
        self.output_len = int(model_args.get("output_len", 12))   # this is H
        self.k = int(model_args.get("token_k", model_args.get("k", 64)))

        self.patch_len = int(model_args.get("patch_len", 3))
        self.stride = int(model_args.get("stride", 4))

        d = int(model_args.get("d_d", model_args.get("d", 64)))
        num_layer = int(model_args.get("num_layer", 3))

        # pad length so divisible by both stride and patch_len
        base = _lcm(self.stride, self.patch_len)
        self.total_pad = int(math.ceil(self.input_len / base) * base - self.input_len)
        self.m_pad = self.input_len + self.total_pad

        self.num_patches = self.m_pad // self.patch_len
        self.ds_len = self.m_pad // self.stride  # length per offset series

        # encoders in Z-space
        self.patch_encoder = PatchEncoderZ(
            patch_len=self.patch_len,
            num_patches=self.num_patches,
            d=d,
            output_len=self.output_len,
            num_layer=num_layer,
        )
        self.downsamp_encoder = DownsampEncoderZ(
            ds_len=self.ds_len,
            stride=self.stride,
            d=d,
            output_len=self.output_len,
            num_layer=num_layer,
        )

        # residual shortcut (stability)
        self.residual = nn.Conv2d(
            in_channels=self.input_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

        # graph-specific U_k (non-persistent buffer)
        self.register_buffer("U_k", None, persistent=False)

    def set_graph_state(self, U_k: torch.Tensor):
        """
        Inject U_k for current dataset.
        U_k: [N, k]
        """
        if U_k.dim() != 2:
            raise ValueError(f"U_k must be 2D [N,k], got {U_k.shape}")
        if U_k.shape[1] != self.k:
            raise ValueError(f"U_k k mismatch: got {U_k.shape[1]} vs model k={self.k}")
        self.U_k = U_k

    def forward(self, history_data: torch.Tensor, future_data: Optional[torch.Tensor] = None,
                batch_seen: int = 0, epoch: int = 0, train: bool = True, **kwargs) -> torch.Tensor:
        """
        history_data: [B, m, k, 1]  (canonical Z, already calibrated)
        return:       [B, H, N, 1]  (node-domain forecast)
        """
        if self.U_k is None:
            raise RuntimeError(
                "U_k is not attached. Call model.set_graph_state(U_k) "
                "when switching dataset in runner."
            )

        Z = history_data  # [B,m,k,1]
        if Z.dim() != 4 or Z.shape[-1] != 1:
            raise ValueError(f"history_data must be [B,m,k,1], got {Z.shape}")

        B, m, k, _ = Z.shape
        if m != self.input_len:
            raise ValueError(f"input_len mismatch: model expects m={self.input_len}, got {m}")
        if k != self.k:
            raise ValueError(f"k mismatch: model expects k={self.k}, got {k}")

        # padding to make divisible by lcm(stride, patch_len)
        if self.total_pad > 0:
            pad_block = Z[:, -1:, :, :].expand(-1, self.total_pad, -1, -1)
            Z_aug = torch.cat([pad_block, Z], dim=1)  # [B,m_pad,k,1]
        else:
            Z_aug = Z  # [B,m,k,1]

        # build downsample input: [B,stride,ds_len,k,1]
        ds_list = [Z_aug[:, i::self.stride, :, :] for i in range(self.stride)]
        ds_input = torch.stack(ds_list, dim=1)

        # build patch input: [B,num_patches,patch_len,k,1]
        patch_input = Z_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        # unfold gives [B, P, patch_len, k, 1]
        # ensure contiguity
        patch_input = patch_input.contiguous()

        patch_pred_k = self.patch_encoder(patch_input)      # [B,H,k,1]
        ds_pred_k = self.downsamp_encoder(ds_input)         # [B,H,k,1]

        # residual shortcut uses [B,m,k,1] -> conv2d in_channels=m
        res_pred_k = self.residual(Z.permute(0, 1, 2, 3))   # already [B,m,k,1] as [B,C,H,W] with C=m
        # NOTE: residual conv expects input [B,in_channels,k,1] so Z is fine:
        # PyTorch Conv2d expects [B,C,H,W], here H=k, W=1, C=m
        # Our Z shape is [B,m,k,1], so OK.

        pred_k = patch_pred_k + ds_pred_k + res_pred_k      # [B,H,k,1]

        if kwargs.get("return_mode", False):
            return pred_k

        # reconstruct to node domain: Y = U_k @ mode_vec
        # pred_k: [B,H,k,1] -> [B,H,k]
        pred_k_s = pred_k.squeeze(-1)
        # U_k: [N,k]
        U = self.U_k.to(pred_k_s.device, dtype=pred_k_s.dtype)
        # [B,H,N] = einsum("nk,bhk->bhn")
        pred_node = torch.einsum("nk,bhk->bhn", U, pred_k_s).unsqueeze(-1)  # [B,H,N,1]

        return pred_node