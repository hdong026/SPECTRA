from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from basicts.archs.arch_zoo.KASA_arch_v2.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.gcn import ABCDSpatialModule

# ==========================================
# 恢复标准 KAN 定义 (本实验只消融空间模块，谱模块要保持最强状态)
# ==========================================
class SimpleKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super(SimpleKANLinear, self).__init__()
        self.grid_size = grid_size
        self.base_linear = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)
        nn.init.kaiming_uniform_(self.base_linear.weight)
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)

    def forward(self, x):
        base = self.base_linear(F.silu(x))
        x_uns = x.unsqueeze(-1)
        basis = torch.exp(-((x_uns - self.grid) / (2 / (self.grid_size - 1))) ** 2)
        spline = torch.einsum("...ig,oig->...o", basis, self.spline_weight)
        return base + spline

class KASA_v2_wo_GCN(nn.Module):
    def __init__(self, **model_args):
        super(KASA_v2_wo_GCN, self).__init__()
        # 参数保存
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        
        # 实验三配置: input_dim 必须是 4 (保持谱注入开启)
        self.input_dim = model_args["input_dim"] 
        
        self.output_len = model_args["output_len"]
        self.patch_len = model_args["patch_len"]
        self.stride = model_args["stride"]
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args["d_td"]
        self.d_dw = model_args["d_dw"]
        self.d_d = model_args["d_d"]
        self.d_spa = model_args["d_spa"]

        self.if_time_in_day = model_args["if_time_in_day"]
        self.if_day_in_week = model_args["if_day_in_week"]
        self.if_spatial = model_args["if_spatial"]
        self.num_layer = model_args["num_layer"]
        self.spatial_scheme = str(model_args.get("spatial_scheme", "legacy")).upper()

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        # 🔥 关键修改 1: 强制关闭所有高级图特性 (Hardcode Disable)
        # 无论 Config 文件里怎么写，这里全部强制设为 False
        print(">>> [KASA Ablation] Experiment 3: FORCING Hybrid/Dynamic/Adaptive Spatial to FALSE <<<")
        print(">>> [KASA Ablation] Model will behave like a Static Graph Model <<<")
        
        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            spatial_scheme=self.spatial_scheme,
            adj_mx_path=model_args.get("adj_mx_path"),
            use_gcn=model_args.get("use_gcn", False), # GCN 基础模块可以留着，但高级特性关掉
            gcn_hidden_dim=model_args.get("gcn_hidden_dim", 64),
            
            # --- 强制关闭区域 ---
            use_dynamic_spatial=False,  # 关
            use_adaptive_adj=False,     # 关
            use_hybrid_graph=False,     # 关 (核心)
            use_lightweight_spatial=False, # 关
            # -------------------
            
            dyn_hidden_dim=model_args.get("dyn_hidden_dim", 64),
            dyn_topk=model_args.get("dyn_topk", 20),
            dyn_tau=model_args.get("dyn_tau", 0.5),
            dyn_alpha=model_args.get("dyn_alpha", 0.15),
            dyn_static_weight=model_args.get("dyn_static_weight", 0.2),
            adp_hidden_dim=model_args.get("adp_hidden_dim", 32),
            adp_topk=model_args.get("adp_topk", 20),
            adp_tau=model_args.get("adp_tau", 0.5),
            adp_alpha=model_args.get("adp_alpha", 0.1),
            hybrid_alpha=model_args.get("hybrid_alpha", 0.2),
            light_alpha=model_args.get("light_alpha", 0.05),
        )

        encoder_input_dim = 3 
        
        self.patch_encoder = PatchEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        self.downsamp_encoder = DownsampEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Main Residual
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
        # 🔥 关键修改 2: 恢复 Spectral KAN
        # 我们希望保留谱特征，只看空间模块的影响
        if self.input_dim > 3:
            self.prior_kan = SimpleKANLinear(1, 1) # 恢复 KAN
            if self.input_len != self.output_len:
                self.time_proj = nn.Linear(self.input_len, self.output_len)
            print(">>> [KASA] Spectral KAN Branch ENABLED (Full Spectral Mode) <<<")
        else:
            print(">>> [WARNING] Input Dim is 3! This is Experiment 1 setting, not Experiment 3! Check Config! <<<")

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # history_data: [B, L, N, 4]
        
        enhanced_spa_emb = self.spatial_module.get_enhanced_spatial_embedding(self.spa_codebook)
        
        main_input = history_data[..., :3]

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat((main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input), dim=1)
        else:
            main_input_aug = main_input

        downsamp_input = [main_input_aug[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3) 

        patch_predict = self.patch_encoder(patch_input, spatial_codebook=enhanced_spa_emb)
        downsamp_predict = self.downsamp_encoder(downsamp_input, spatial_codebook=enhanced_spa_emb)

        res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
        res_out = self.residual(res_input)

        output = patch_predict + downsamp_predict + res_out

        # Spatial Refinement (Static Only)
        # 这里调用的 refine_prediction 内部会因为 use_hybrid_graph=False 而只执行基础 GCN 或 Static Graph 逻辑
        history_flow = history_data[..., 0]
        output = self.spatial_module.refine_prediction(output, history_flow)
        
        # Spectral KAN Injection (Keep Enabled)
        if self.input_dim > 3:
            prior_data = history_data[..., 3:4]
            kan_prior = self.prior_kan(prior_data)
            
            if self.input_len != self.output_len:
                kan_prior = kan_prior.permute(0, 2, 3, 1)
                kan_prior = self.time_proj(kan_prior)
                kan_prior = kan_prior.permute(0, 3, 1, 2)
            
            output = output + kan_prior

        return output