"""
KASA_TKDE (SPECTRA-GFM ready, Prior-optional)

Key changes vs your earlier version:
  1) Topology-agnostic spatial state:
     - No N×d spa_codebook Parameter
     - spa_codebook = U_k(G) @ Z_spa, where:
         U_k(G): graph-specific structural basis (buffer/token, persistent=False)
         Z_spa : shared learnable coefficients (k×d_spa)
     - U_k is attached by runner via set_graph_state(U_k, eigvals_k)

  2) Long-horizon/multi-frequency priors:
     - Default recommendation: DO NOT require any long-history prior at inference/PEFT.
     - Instead, during pretraining you can enable an auxiliary multi-frequency loss
       (spectral consistency) that uses only (prediction, future ground-truth flow).

     - Optional (only for ablation / teacher / legacy): prior-injection branch can be enabled
       if you still feed a 4th channel and set enable_prior_injection=True.

  3) Pretrain auxiliary mode:
     - When forward(..., pretrain_aux=True, train=True), returns dict:
         {"pred": output, "aux_loss": lambda_spec * loss_spec}
     - Otherwise returns output tensor as usual.
"""

from math import ceil
import torch
from torch import nn
import torch.nn.functional as F

from basicts.archs.arch_zoo.KASA_arch_v2.kasa_components_tkde import (
    PatchEncoder,
    DownsampEncoder,
    ABCDSpatialModule,
)


# =========================================================================
# Koopman-Arnold Evolution Operator (optional / legacy prior-injection branch)
# =========================================================================
class TimeAwareKAN(nn.Module):
    def __init__(self, in_features: int, out_features: int, horizon: int, grid_size: int = 5):
        super().__init__()
        self.grid_size = grid_size
        self.horizon = horizon

        self.time_emb = nn.Embedding(horizon, 8)
        self.base_linear = nn.Linear(in_features + 8, out_features)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features + 8, grid_size))
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)

        nn.init.kaiming_uniform_(self.base_linear.weight)
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, N, C]
        t_indices: [H]  (0..H-1)
        """
        B, H, N, C = x.shape
        assert H == t_indices.numel(), "KAN expects horizon dimension to match t_indices length."

        t_emb = self.time_emb(t_indices).unsqueeze(0).unsqueeze(2).expand(B, H, N, -1)
        x_in = torch.cat([x, t_emb], dim=-1)  # [B,H,N,C+8]

        base = self.base_linear(F.silu(x_in))

        x_uns = x_in.unsqueeze(-1)
        basis = torch.exp(-((x_uns - self.grid) / (2 / (self.grid_size - 1))) ** 2)
        spline = torch.einsum("...ig,oig->...o", basis, self.spline_weight)

        return base + spline


def _band_energy(x: torch.Tensor, bands=(1, 2, 3, 4)) -> torch.Tensor:
    """
    Multi-frequency target: band power at selected frequency bins.

    x: [B, H, N, 1] (normalized flow)
    return: [B, len(bands), N]
    """
    x = x.squeeze(-1)               # [B,H,N]
    X = torch.fft.rfft(x, dim=1)    # [B,F,N], F=H//2+1
    P = X.real.pow(2) + X.imag.pow(2)

    outs = []
    F_bins = P.shape[1]
    for b in bands:
        bb = int(b)
        bb = max(0, min(bb, F_bins - 1))
        outs.append(P[:, bb, :])    # [B,N]
    return torch.stack(outs, dim=1) # [B,Bands,N]


# =========================================================================
# Main Model: KASA_TKDE (SPECTRA-GFM)
# =========================================================================
class KASA_TKDE(nn.Module):
    def __init__(self, **model_args):
        super().__init__()

        # ----------------------------
        # 1) Configuration
        # ----------------------------
        self.node_size = int(model_args["node_size"])
        self.input_len = int(model_args["input_len"])
        self.output_len = int(model_args["output_len"])
        self.input_dim = int(model_args["input_dim"])  # 3 (recommended) or 4 (if you still feed prior for ablations)
        self.patch_len = int(model_args["patch_len"])
        self.stride = int(model_args["stride"])

        self.td_size = int(model_args["td_size"])
        self.dw_size = int(model_args["dw_size"])
        self.d_td = int(model_args["d_td"])
        self.d_dw = int(model_args["d_dw"])
        self.d_d = int(model_args["d_d"])
        self.d_spa = int(model_args["d_spa"])
        self.num_layer = int(model_args["num_layer"])

        # topology-agnostic structural basis size
        self.token_k = int(model_args.get("token_k", 64))

        # flags
        self.if_time_in_day = bool(model_args["if_time_in_day"])
        self.if_day_in_week = bool(model_args["if_day_in_week"])
        self.if_spatial = bool(model_args["if_spatial"])

        # Optional legacy prior injection (should be OFF by default for GFM story)
        # Enable only when you explicitly want to test prior-injection ablations.
        self.enable_prior_injection = bool(model_args.get("enable_prior_injection", False))

        # ----------------------------
        # 2) Learnable temporal embeddings (unchanged)
        # ----------------------------
        self.td_codebook = None
        self.dw_codebook = None

        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)

        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)

        # ----------------------------
        # 3) SPECTRA-GFM: graph state buffers + topology-agnostic params
        # ----------------------------
        # Graph structural basis U_k(G): [N,k] (graph-specific, not saved in ckpt)
        self.register_buffer("U_k", None, persistent=False)
        self.register_buffer("eigvals_k", None, persistent=False)  # optional for diagnostics / conditioning later

        # Shared learnable coefficients Z_spa: [k, d_spa]
        self.Z_spa = None
        if self.if_spatial:
            self.Z_spa = nn.Parameter(torch.empty(self.token_k, self.d_spa))
            nn.init.xavier_uniform_(self.Z_spa)

        # ----------------------------
        # 4) Core Modules (Phase I & II backbone)
        # ----------------------------
        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            spatial_scheme=str(model_args.get("spatial_scheme", "legacy")).upper(),
            adj_mx_path=model_args.get("adj_mx_path"),

            use_gcn=model_args.get("use_gcn", False),
            gcn_hidden_dim=model_args.get("gcn_hidden_dim", 64),

            use_dynamic_spatial=model_args.get("use_dynamic_spatial", False),
            dyn_hidden_dim=model_args.get("dyn_hidden_dim", 64),
            dyn_topk=model_args.get("dyn_topk", 20),
            dyn_tau=model_args.get("dyn_tau", 0.5),
            dyn_alpha=model_args.get("dyn_alpha", 0.15),
            dyn_static_weight=model_args.get("dyn_static_weight", 0.2),

            use_adaptive_adj=model_args.get("use_adaptive_adj", False),
            adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
            adp_topk=model_args.get("adp_topk", 20),
            adp_tau=model_args.get("adp_tau", 0.5),
            adp_alpha=model_args.get("adp_alpha", 0.1),

            use_hybrid_graph=model_args.get("use_hybrid_graph", False),
            hybrid_alpha=model_args.get("hybrid_alpha", 0.2),

            use_lightweight_spatial=model_args.get("use_lightweight_spatial", False),
            light_alpha=model_args.get("light_alpha", 0.05),

            # Stage-2: pass token_k so spatial module can use U_k@Z for adaptive adjacency (B2)
            token_k=self.token_k,
        )

        # Encoders only consume flow/tod/dow (3 channels)
        encoder_input_dim = 3

        self.patch_encoder = PatchEncoder(
            self.td_size, self.dw_size,
            self.td_codebook, self.dw_codebook,
            None,  # spa_codebook kept for signature compatibility, not stored
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            encoder_input_dim, self.patch_len, self.stride,
            self.d_d, self.d_td, self.d_dw, self.d_spa,
            self.output_len, self.num_layer
        )

        self.downsamp_encoder = DownsampEncoder(
            self.td_size, self.dw_size,
            self.td_codebook, self.dw_codebook,
            None,  # spa_codebook kept for signature compatibility, not stored
            self.if_time_in_day, self.if_day_in_week, self.if_spatial,
            encoder_input_dim, self.patch_len, self.stride,
            self.d_d, self.d_td, self.d_dw, self.d_spa,
            self.output_len, self.num_layer
        )

        self.residual = nn.Conv2d(
            in_channels=self.input_len, out_channels=self.output_len,
            kernel_size=(1, 1), bias=True
        )

        # ----------------------------
        # 5) Optional legacy Phase-III (prior injection). OFF by default.
        # ----------------------------
        self.freq_modulator = None
        self.spectral_evolver = None
        self.spectral_gate = None
        if self.enable_prior_injection:
            # expect prior in channel-3; horizon-aligned
            self.freq_modulator = nn.Parameter(torch.ones(1, self.output_len, 1, 1))
            self.spectral_evolver = TimeAwareKAN(in_features=1, out_features=1, horizon=self.output_len)
            self.spectral_gate = nn.Parameter(torch.tensor([0.1]))

    # ----------------------------
    # External API: attach graph state
    # ----------------------------
    def set_graph_state(self, U_k: torch.Tensor, eigvals_k: torch.Tensor = None):
        """
        Attach graph-specific structural basis to the model.
        U_k: [N,k], eigvals_k: optional [k]
        """
        self.U_k = U_k
        if eigvals_k is not None:
            self.eigvals_k = eigvals_k

        # also attach to spatial module (needed for B2 adaptive adjacency)
        if hasattr(self.spatial_module, "set_structural_basis"):
            self.spatial_module.set_structural_basis(U_k)

    def _make_spa_codebook(self, device, dtype):
        """Generate topology-agnostic spatial codebook: [N,d_spa] = U_k @ Z_spa."""
        if not self.if_spatial:
            return None
        assert self.U_k is not None, "U_k must be attached via set_graph_state() before forward."
        U = self.U_k.to(device=device, dtype=dtype)
        Z = self.Z_spa.to(device=device, dtype=dtype)
        return U @ Z  # [N, d_spa]

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs
    ) -> torch.Tensor:
        """
        history_data: [B, L, N, C_in]
          Recommended: C_in=3 (flow,tod,dow). If C_in=4, the 4th channel is treated as "prior" only for ablation.

        future_data: [B, H, N, C_in] (runner passes selected forward features too)
        """

        # --------------- Phase I & II: backbone ---------------
        # Use only the first 3 channels for backbone modeling (flow,tod,dow)
        main_input = history_data[..., :3]  # [B,L,N,3]

        # topology-agnostic spatial codebook
        spa_codebook = self._make_spa_codebook(main_input.device, main_input.dtype) if self.if_spatial else None
        enhanced = self.spatial_module.get_enhanced_spatial_embedding(spa_codebook)
        spatial_codebook_for_enc = enhanced if enhanced is not None else spa_codebook

        # patching logic
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat(
                (main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input), dim=1
            )
        else:
            main_input_aug = main_input

        # downsample input: [B, stride, L/stride, N, 3]
        downsamp_input = [main_input_aug[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        # patch input: [B, P, patch_len, N, 3]
        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3)

        # encoders (spatial_codebook is mandatory when if_spatial=True)
        patch_predict = self.patch_encoder(patch_input, spatial_codebook=spatial_codebook_for_enc)
        downsamp_predict = self.downsamp_encoder(downsamp_input, spatial_codebook=spatial_codebook_for_enc)

        # residual branch on flow only
        res_input = history_data[..., 0:1]  # [B,L,N,1]
        res_out = self.residual(res_input)  # [B,H,N,1]

        if kwargs.get("return_backbone", False):
            return patch_predict + downsamp_predict

        # backbone output
        output = patch_predict + downsamp_predict + res_out  # [B,H,N,1]

        # spatial refinement (dynamic/adaptive/hybrid)
        history_flow = history_data[..., 0]  # [B,L,N]
        output = self.spatial_module.refine_prediction(
            output,
            history_flow,
            dyn_alpha=kwargs.get("dyn_alpha", None),
            adp_alpha=kwargs.get("adp_alpha", None),
            hybrid_alpha=kwargs.get("hybrid_alpha", None),
            dyn_static_weight=kwargs.get("dyn_static_weight", None),
        )

        # --------------- Pretrain auxiliary (recommended) ---------------
        # Multi-frequency spectral consistency: does NOT require any long-history prior.
        if kwargs.get("pretrain_aux", False) and train:
            # future_data may be [B,H,N,3] if USE_PRIOR=0; channel-0 is flow
            y_true = future_data[..., 0:1]
            y_pred = output

            bands = kwargs.get("spec_bands", (1, 2, 3, 4))
            tgt = _band_energy(y_true, bands=bands)
            pred = _band_energy(y_pred, bands=bands)

            loss_spec = torch.mean((pred - tgt) ** 2)
            lam = float(kwargs.get("lambda_spec", 0.5))
            return {"pred": output, "aux_loss": lam * loss_spec}

        # --------------- Optional legacy prior injection (ablation only) ---------------
        # Only active if enable_prior_injection=True AND a 4th channel is present.
        if self.enable_prior_injection and (history_data.shape[-1] >= 4 or (future_data is not None and future_data.shape[-1] >= 4)):
            prior_spec = None
            # Prefer future prior aligned with horizon
            if future_data is not None and future_data.shape[1] == self.output_len and future_data.shape[-1] >= 4:
                prior_spec = future_data[..., 3:4]
            elif history_data.shape[-1] >= 4:
                prior_spec = history_data[:, -self.output_len:, :, 3:4]

            if prior_spec is not None:
                if self.freq_modulator is None or self.freq_modulator.shape[1] != self.output_len:
                    self.freq_modulator = nn.Parameter(torch.ones(1, self.output_len, 1, 1, device=output.device, dtype=output.dtype))

                modulated = prior_spec * self.freq_modulator
                t_indices = torch.arange(self.output_len, device=output.device)
                spec_residual = self.spectral_evolver(modulated, t_indices)
                output = output + self.spectral_gate * spec_residual

        return output