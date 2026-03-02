"""
Merged components: MultiLayerPerceptron, GCN/spatial, PatchEncoder, DownsampEncoder.

[PATCH v1] Stage-1:
  (1) avoid saving adj_mx in ckpt (persistent=False)
  (2) avoid re-registering spa_codebook inside Patch/Downsamp encoders
  (3) make spatial_codebook mandatory when if_spatial=True to prevent silent fallback

[PATCH v2] Stage-2 (Topology-agnostic / SPECTRA-GFM):
  (4) replace N×H adaptive_src/adaptive_dst with U_k(G) @ Z (k×H), where:
        - U_k(G): graph-specific structural basis (buffer/token, persistent=False)
        - Z: shared learnable parameters (topology-agnostic, transferable)
  (5) add set_structural_basis(U_k) API for runner/model to attach graph state
  (6) allow optional conditioning overrides for refine_prediction alphas/weights
"""

from math import ceil, sqrt
import os
import pickle

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ---------------------------------------------------------------------------
# From mlp.py
# ---------------------------------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """input_data: [B, D, P, N]"""
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = self.norm((hidden + input_data).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden


# ---------------------------------------------------------------------------
# From gcn.py
# ---------------------------------------------------------------------------
def normalize_adj(adj):
    """Symmetric normalization D^{-1/2}(A+I)D^{-1/2}."""
    adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def row_normalize(adj):
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / row_sum


def apply_adj(x, adj):
    """Graph propagation on temporal features.
    x: [B, T, N]
    adj: [N, N] or [B, N, N]
    """
    if adj.dim() == 2:
        return torch.einsum("ij,btj->bti", adj, x)
    return torch.einsum("bij,btj->bti", adj, x)


def mask_topk(logits, topk):
    if not (0 < topk < logits.shape[-1]):
        return logits
    topk_index = torch.topk(logits, k=topk, dim=-1).indices
    keep_mask = torch.zeros_like(logits, dtype=torch.bool)
    keep_mask.scatter_(-1, topk_index, True)
    return logits.masked_fill(~keep_mask, float("-inf"))


def load_adj_from_pickle(adj_mx_path):
    if not adj_mx_path or not os.path.exists(adj_mx_path):
        return None
    with open(adj_mx_path, "rb") as f:
        try:
            adj_obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            adj_obj = pickle.load(f, encoding="latin1")
    if isinstance(adj_obj, (list, tuple)):
        for item in reversed(adj_obj):
            if hasattr(item, "shape") and len(item.shape) == 2:
                adj_obj = item
                break
    adj_tensor = torch.as_tensor(adj_obj, dtype=torch.float32)
    return normalize_adj(adj_tensor)


class GCNLayer(nn.Module):
    """Single GCN layer H = ReLU(A X W)."""

    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activate = nn.ReLU()

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.mm(adj, support)
        return self.activate(output)


class ABCDSpatialModule(nn.Module):
    """Unified spatial module implementing schemes A/B/C/D.

    Stage-2 (Topology-agnostic) notes:
      - adaptive adjacency no longer uses N×H learnable tables.
      - Instead: src = U_k @ Z_src, dst = U_k @ Z_dst
        where U_k is a graph-specific buffer set externally via set_structural_basis().
    """

    def __init__(
        self,
        node_size,
        input_len,
        d_spa,
        if_spatial,
        spatial_scheme="LEGACY",
        adj_mx_path=None,
        use_gcn=False,
        gcn_hidden_dim=64,
        use_dynamic_spatial=False,
        dyn_hidden_dim=64,
        dyn_topk=20,
        dyn_tau=0.5,
        dyn_alpha=0.15,
        dyn_static_weight=0.2,
        use_adaptive_adj=False,
        adp_hidden_dim=32,
        adp_topk=20,
        adp_tau=0.5,
        adp_alpha=0.1,
        use_hybrid_graph=False,
        hybrid_alpha=0.2,
        use_lightweight_spatial=False,
        light_alpha=0.05,
        # [NEW] topology-agnostic basis size
        token_k=64,
    ):
        super().__init__()
        self.node_size = node_size
        self.input_len = input_len
        self.d_spa = d_spa
        self.if_spatial = if_spatial
        self.spatial_scheme = str(spatial_scheme).upper()
        self.token_k = int(token_k)

        # Control flags (explicit flags + scheme override).
        self.use_gcn = use_gcn
        self.use_dynamic_spatial = use_dynamic_spatial
        self.use_adaptive_adj = use_adaptive_adj
        self.use_hybrid_graph = use_hybrid_graph
        self.use_lightweight_spatial = use_lightweight_spatial

        if self.spatial_scheme in {"A", "B", "C", "D"}:
            self.use_gcn = self.spatial_scheme in {"A", "C"}
            self.use_dynamic_spatial = self.spatial_scheme in {"B", "C"}
            self.use_adaptive_adj = self.spatial_scheme in {"C"}
            self.use_hybrid_graph = self.spatial_scheme in {"C"}
            self.use_lightweight_spatial = self.spatial_scheme in {"D"}
        if self.use_hybrid_graph:
            self.use_dynamic_spatial = True
            self.use_adaptive_adj = True

        # Hyper-params.
        self.dyn_hidden_dim = dyn_hidden_dim
        self.dyn_topk = dyn_topk
        self.dyn_tau = dyn_tau
        self.dyn_alpha = dyn_alpha
        self.dyn_static_weight = dyn_static_weight

        self.adp_hidden_dim = adp_hidden_dim
        self.adp_topk = adp_topk
        self.adp_tau = adp_tau
        self.adp_alpha = adp_alpha

        self.hybrid_alpha = hybrid_alpha
        self.light_alpha = light_alpha

        # -----------------------------
        # [PATCH v1] Static adjacency buffer (non-persistent).
        # -----------------------------
        self.register_buffer("adj_mx", None, persistent=False)

        need_static_adj = (
            self.use_gcn
            or self.use_dynamic_spatial
            or self.use_lightweight_spatial
            or self.use_hybrid_graph
        )
        if need_static_adj:
            self.adj_mx = load_adj_from_pickle(adj_mx_path)

        # -----------------------------
        # [PATCH v2] Structural basis buffer U_k (non-persistent).
        # This should be attached externally by runner/model.
        # Shape: [N, k]
        # -----------------------------
        self.register_buffer("U_k", None, persistent=False)

        # Scheme A: static GCN on spatial codebook.
        self.gcn1 = None
        self.gcn2 = None
        if self.use_gcn and self.adj_mx is not None and self.if_spatial:
            self.gcn1 = GCNLayer(self.d_spa, gcn_hidden_dim)
            self.gcn2 = GCNLayer(gcn_hidden_dim, self.d_spa)

        # Scheme B: dynamic graph from flow windows.
        self.dynamic_query = None
        self.dynamic_key = None
        if self.use_dynamic_spatial:
            self.dynamic_query = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)
            self.dynamic_key = nn.Linear(self.input_len, self.dyn_hidden_dim, bias=False)

        # Scheme C (Topology-agnostic): adaptive graph parameters via U_k @ Z.
        # NOTE: No N×H parameters.
        self.Z_src = None
        self.Z_dst = None
        if self.use_adaptive_adj:
            self.Z_src = nn.Parameter(torch.empty(self.token_k, self.adp_hidden_dim))
            self.Z_dst = nn.Parameter(torch.empty(self.token_k, self.adp_hidden_dim))
            nn.init.xavier_uniform_(self.Z_src)
            nn.init.xavier_uniform_(self.Z_dst)

        self.hybrid_logits = None
        if self.use_hybrid_graph:
            self.hybrid_logits = nn.Parameter(torch.zeros(3))

    # -----------------------------
    # External API: attach structural basis
    # -----------------------------
    def set_structural_basis(self, U_k: torch.Tensor):
        """Attach graph structural basis.
        Args:
            U_k: [N, k] tensor on the correct device/dtype (or will be cast by caller).
        """
        self.U_k = U_k

    def set_static_adj(self, adj_mx: torch.Tensor):
        self.adj_mx = adj_mx

    def _build_dynamic_adj(self, history_flow):
        node_signal = history_flow.permute(0, 2, 1)  # [B, N, L]
        query = F.normalize(self.dynamic_query(node_signal), p=2, dim=-1)
        key = F.normalize(self.dynamic_key(node_signal), p=2, dim=-1)
        logits = torch.matmul(query, key.transpose(-1, -2)) / sqrt(self.dyn_hidden_dim)

        if self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx)
            logits = logits + self.dyn_static_weight * static_adj.unsqueeze(0)

        logits = mask_topk(logits, self.dyn_topk)
        dyn_adj = torch.softmax(logits / max(self.dyn_tau, 1e-6), dim=-1)
        return dyn_adj

    def _build_adaptive_adj(self):
        # topology-agnostic adaptive adjacency: src/dst are generated from U_k @ Z_*
        assert self.U_k is not None, "U_k must be set via set_structural_basis() for adaptive adjacency."
        # U_k: [N,k], Z_*: [k,H] => [N,H]
        src = F.normalize(self.U_k @ self.Z_src, p=2, dim=-1)
        dst = F.normalize(self.U_k @ self.Z_dst, p=2, dim=-1)
        logits = torch.matmul(src, dst.transpose(0, 1)) / sqrt(self.adp_hidden_dim)
        logits = mask_topk(logits, self.adp_topk)
        adp_adj = torch.softmax(logits / max(self.adp_tau, 1e-6), dim=-1)
        return adp_adj

    def _build_hybrid_adj(self, history_flow):
        batch_size = history_flow.shape[0]
        dynamic_adj = self._build_dynamic_adj(history_flow)
        adaptive_adj = self._build_adaptive_adj().unsqueeze(0).expand(batch_size, -1, -1)

        if self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            static_adj = (
                torch.eye(self.node_size, device=history_flow.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )

        weights = torch.softmax(self.hybrid_logits, dim=0)
        return weights[0] * static_adj + weights[1] * adaptive_adj + weights[2] * dynamic_adj

    def get_enhanced_spatial_embedding(self, spa_codebook):
        """Scheme A/C: enhance spatial codebook before encoders."""
        if not self.if_spatial or spa_codebook is None:
            return None
        if self.use_gcn and self.adj_mx is not None and self.gcn1 is not None:
            emb = self.gcn1(spa_codebook, self.adj_mx)
            emb = self.gcn2(emb, self.adj_mx)
            return emb
        return None

    def refine_prediction(
        self,
        output,
        history_flow,
        # optional conditioning overrides (scalars)
        dyn_alpha=None,
        adp_alpha=None,
        hybrid_alpha=None,
        dyn_static_weight=None,
        light_alpha=None,
    ):
        """Apply B/C/D or adaptive-only refinement on prediction output.

        Args:
            output: [B, T, N, 1]
            history_flow: [B, L, N]
            dyn_alpha/adp_alpha/hybrid_alpha: override default weights if not None
            dyn_static_weight: override static injection weight into dynamic logits if not None
            light_alpha: override lightweight smoothing alpha if not None
        """
        x = output.squeeze(-1)

        # defaults
        dyn_alpha = self.dyn_alpha if dyn_alpha is None else dyn_alpha
        adp_alpha = self.adp_alpha if adp_alpha is None else adp_alpha
        hybrid_alpha = self.hybrid_alpha if hybrid_alpha is None else hybrid_alpha
        light_alpha = self.light_alpha if light_alpha is None else light_alpha
        if dyn_static_weight is not None:
            self.dyn_static_weight = dyn_static_weight

        if self.use_hybrid_graph:
            hybrid_adj = self._build_hybrid_adj(history_flow)
            refine = apply_adj(x, hybrid_adj).unsqueeze(-1)
            return output + hybrid_alpha * refine

        if self.use_dynamic_spatial:
            dyn_adj = self._build_dynamic_adj(history_flow)
            refine = apply_adj(x, dyn_adj).unsqueeze(-1)
            return output + dyn_alpha * refine

        if self.use_adaptive_adj:
            adp_adj = self._build_adaptive_adj()
            refine = apply_adj(x, adp_adj).unsqueeze(-1)
            return output + adp_alpha * refine

        if self.use_lightweight_spatial and self.adj_mx is not None:
            static_adj = row_normalize(self.adj_mx)
            smooth = apply_adj(x, static_adj).unsqueeze(-1) - output
            return output + light_alpha * smooth

        return output


# ---------------------------------------------------------------------------
# Patch Encoder
# ---------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    def __init__(
        self,
        td_size,
        dw_size,
        td_codebook,
        dw_codebook,
        spa_codebook,  # kept for signature compatibility (not stored)
        if_time_in_day,
        if_day_in_week,
        if_spatial,
        input_dim,
        patch_len,
        stride,
        d_d,
        d_td,
        d_dw,
        d_spa,
        output_len,
        num_layer,
    ):
        super(PatchEncoder, self).__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook

        # [PATCH v1] Do NOT hold spa_codebook Parameter here (avoid duplicated registration).
        self.spa_codebook = None

        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.dw_size = dw_size
        self.stride = stride

        self.data_embedding_layer = nn.Conv2d(
            in_channels=input_dim * patch_len, out_channels=d_d, kernel_size=(1, 1), bias=True
        )
        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2

        self.temporal_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim + d_spa * int(self.if_spatial),
                    self.hidden_dim + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.spatial_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    d_d + d_spa * int(self.if_spatial),
                    d_d + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.data_encoder = nn.Sequential(*[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)])

        self.projection1 = nn.Conv2d(
            in_channels=(self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, patch_input, spatial_codebook=None):
        # B P L N C
        batch_size, num, _, _, _ = patch_input.shape

        # Temporal Embedding
        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb, day_in_week_end_emb, future_day_in_week_emb = None, None, None

        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]
            time_start_idx = torch.clamp((time_in_day_data[:, :, 0, :] * self.td_size).long(), 0, self.td_size - 1)
            time_end_idx = torch.clamp((time_in_day_data[:, :, -1, :] * self.td_size).long(), 0, self.td_size - 1)
            time_in_day_start_emb = self.td_codebook[time_start_idx]
            time_in_day_end_emb = self.td_codebook[time_end_idx]
            future_time_idx = ((time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size).long()
            future_time_idx = torch.clamp(future_time_idx, 0, self.td_size - 1)
            future_time_in_day_emb = self.td_codebook[future_time_idx].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb, time_in_day_end_emb, future_time_in_day_emb = None, None, None

        # Spatial Embedding
        if self.if_spatial:
            # [PATCH v1] Make spatial_codebook mandatory to avoid fallback to a registered Parameter.
            assert spatial_codebook is not None, "PatchEncoder requires spatial_codebook when if_spatial=True"
            spatial_emb = (
                spatial_codebook.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .unsqueeze(1)
                .expand(-1, num, -1, -1)
            )
        else:
            spatial_emb = None

        # time series embedding
        data_emb = self.data_embedding_layer(
            torch.concat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=2).permute(0, 2, 1, 3)
        ).permute(0, 2, 3, 1)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # spatial encoding
        if self.if_spatial:
            hidden_input = torch.concat((data_emb, spatial_emb), dim=-1)
        else:
            hidden_input = data_emb
        hidden = hidden_input.permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)

        # temporal encoding
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)

        hidden = rearrange(hidden, "B D P N -> B (D P) N").unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        predict = self.projection1(hidden)
        return predict


# ---------------------------------------------------------------------------
# Downsamp Encoder
# ---------------------------------------------------------------------------
class DownsampEncoder(nn.Module):
    def __init__(
        self,
        td_size,
        dw_size,
        td_codebook,
        dw_codebook,
        spa_codebook,  # kept for signature compatibility (not stored)
        if_time_in_day,
        if_day_in_week,
        if_spatial,
        input_dim,
        patch_len,
        stride,
        d_d,
        d_td,
        d_dw,
        d_spa,
        output_len,
        num_layer,
    ):
        super(DownsampEncoder, self).__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook

        # [PATCH v1] Do NOT hold spa_codebook Parameter here (avoid duplicated registration).
        self.spa_codebook = None

        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.dw_size = dw_size
        self.stride = stride

        self.data_embedding_layer = nn.Conv2d(
            in_channels=input_dim * patch_len, out_channels=d_d, kernel_size=(1, 1), bias=True
        )
        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2

        self.temporal_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim + d_spa * int(self.if_spatial),
                    self.hidden_dim + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.spatial_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    d_d + d_spa * int(self.if_spatial),
                    d_d + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )

        self.data_encoder = nn.Sequential(*[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)])

        self.projection1 = nn.Conv2d(
            in_channels=(self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, patch_input, spatial_codebook=None):
        batch_size, num, _, _, _ = patch_input.shape

        # Temporal Embedding
        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]
            time_start_idx = torch.clamp((time_in_day_data[:, :, 0, :] * self.td_size).long(), 0, self.td_size - 1)
            time_end_idx = torch.clamp((time_in_day_data[:, :, -1, :] * self.td_size).long(), 0, self.td_size - 1)
            time_in_day_start_emb = self.td_codebook[time_start_idx]
            time_in_day_end_emb = self.td_codebook[time_end_idx]
            future_time_idx = ((time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size).long()
            future_time_idx = torch.clamp(future_time_idx, 0, self.td_size - 1)
            future_time_in_day_emb = self.td_codebook[future_time_idx].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb, time_in_day_end_emb, future_time_in_day_emb = None, None, None

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb, day_in_week_end_emb, future_day_in_week_emb = None, None, None

        # Spatial Embedding
        if self.if_spatial:
            # [PATCH v1] Make spatial_codebook mandatory.
            assert spatial_codebook is not None, "DownsampEncoder requires spatial_codebook when if_spatial=True"
            spatial_emb = (
                spatial_codebook.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .unsqueeze(1)
                .expand(-1, num, -1, -1)
            )
        else:
            spatial_emb = None

        # time series embedding
        data_emb = self.data_embedding_layer(
            torch.concat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=2).permute(0, 2, 1, 3)
        ).permute(0, 2, 3, 1)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # spatial encoding
        if self.if_spatial:
            hidden_input = torch.concat((data_emb, spatial_emb), dim=-1)
        else:
            hidden_input = data_emb
        hidden = hidden_input.permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)

        # temporal encoding
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)

        hidden = rearrange(hidden, "B D P N -> B (D P) N").unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        predict = self.projection1(hidden)
        return predict