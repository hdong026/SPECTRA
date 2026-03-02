import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint

from basicts.archs.arch_zoo.KASA_arch_v2.KASA_arch import KASA_v2
from .utils import GraphSpectralUtils, SpectralKANVectorField


class LatentSpectralKAFM(nn.Module):
    def __init__(self, cfgs, kasa_backbone_args):
        super().__init__()

        self.node_size = kasa_backbone_args['node_size']

        # 1) KASA backbone
        self.kasa_model = KASA_v2(**kasa_backbone_args)

        # 2) Adapter (zero init)
        self.kasa_adapter = nn.Conv2d(
            in_channels=kasa_backbone_args['output_len'],
            out_channels=kasa_backbone_args['output_len'],
            kernel_size=1
        )
        nn.init.zeros_(self.kasa_adapter.weight)
        nn.init.zeros_(self.kasa_adapter.bias)

        # 3) Safe gate
        self.kasa_gate = nn.Parameter(torch.zeros(1))

        # 4) Spectral utils (NO hardcoded cuda)
        self.spec_utils = GraphSpectralUtils(
            adj_mx_path=kasa_backbone_args['adj_mx_path'],
            device=None,
            top_k=cfgs.get('top_k'),
            node_size=kasa_backbone_args['node_size']
        )

        # 5) KAN vector field
        self.vector_field = SpectralKANVectorField(
            input_dim=1,
            hidden_dim=cfgs['hidden_dim'],
            cond_dim=1,
            grid_size=cfgs['grid_size']
        )

        # 6) Flow matching gate
        self.fm_gate = nn.Parameter(torch.zeros(1))

    def forward(self, history_data, future_data=None, batch_seen=None, epoch=None, train=True, **kwargs):
        # Scale Factor
        scale_factor = torch.sqrt(torch.tensor(self.node_size, device=history_data.device, dtype=torch.float32))

        # 1) Prior baseline: repeat last step
        prior_last_step = history_data[:, -1:, :, 3:4]
        prior_baseline = prior_last_step.expand(-1, 12, -1, -1)

        # 2) KASA backbone
        kasa_raw = self.kasa_model(history_data, None, batch_seen, epoch, train)
        kasa_in = kasa_raw.permute(0, 1, 2, 3)

        # adapter -> tanh clamp -> gate
        adapter_out = self.kasa_adapter(kasa_in)
        kasa_contribution = torch.tanh(adapter_out) * self.kasa_gate

        # 3) base prediction
        base_pred = prior_baseline + kasa_contribution

        # 4) spectral condition (as tokens, keep your current behavior)
        cond_spec_raw = self.spec_utils.gft(base_pred)
        cond_spec = cond_spec_raw / scale_factor

        if train and future_data is not None:
            target = future_data[..., 0:1]
            residual = target - base_pred.detach()

            x_1_raw = self.spec_utils.gft(residual)
            x_1 = x_1_raw / scale_factor

            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=x_1.device).view(-1, 1, 1, 1)
            x_t = (1 - t) * x_0 + t * x_1
            u_t = x_1 - x_0

            v_t = self.vector_field(x_t, t.flatten(), cond_spec)
            return F.mse_loss(v_t, u_t)

        # inference
        B, L, N, C = cond_spec.shape
        if torch.abs(self.fm_gate) < 1e-6:
            return base_pred

        ode_func = lambda t, x: self.vector_field(x, t.repeat(B), cond_spec)
        x_0 = torch.randn(B, L, N, C).to(base_pred.device)
        trajectory = odeint(
            ode_func,
            x_0,
            torch.tensor([0., 1.]).to(base_pred.device),
            method='euler',
            options={'step_size': 0.1}
        )
        gen_residual_spec = trajectory[-1]
        gen_residual_spec = gen_residual_spec * scale_factor
        gen_residual = self.spec_utils.igft(gen_residual_spec)
        return base_pred + self.fm_gate * gen_residual