from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from basicts.archs.arch_zoo.KASA_arch_v2.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.downsamp_emb import DownsampEncoder
from basicts.archs.arch_zoo.KASA_arch_v2.gcn import ABCDSpatialModule

# ==========================================
# 定义 SimpleMLP (用于替换 KAN)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        # 一个标准的 2层 MLP: Linear -> ReLU -> Linear
        # hidden_dim 设为 64，保证参数量甚至略多于 KAN (grid=5时参数很少)，
        # 以防止审稿人说 MLP 输是因为参数太少。
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        
        # 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class KASA_v2_wo_KAN(nn.Module):
    def __init__(self, **model_args):
        super(KASA_v2_wo_KAN, self).__init__()
        # 参数保存
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        
        # 🔥 关键点 1: 读取配置
        # 实验二配置: "input_dim": 4 (我们需要第4维数据，但用 MLP 处理)
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

        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            spatial_scheme=self.spatial_scheme,
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
        )

        # 🔥 关键点 2: 依然只传 3 通道给主干
        encoder_input_dim = 3 
        
        self.patch_encoder = PatchEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        self.downsamp_encoder = DownsampEncoder(self.td_size, self.dw_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          encoder_input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Main Residual
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
        # 🔥 关键点 3: 初始化 MLP 而不是 KAN
        if self.input_dim > 3:
            # 这里的输入输出依然是 1维 -> 1维
            self.prior_kan = SimpleMLP(in_features=1, out_features=1) 
            
            # 同样需要时间对齐层
            if self.input_len != self.output_len:
                self.time_proj = nn.Linear(self.input_len, self.output_len)
            
            print(">>> [KASA Ablation] Experiment 2: RBF-KAN replaced by SimpleMLP <<<")
        else:
            # 如果配置错了写成了3，这里会报错提醒你
            print(">>> [WARNING] Input Dim is 3! MLP will NOT be used. Check your Config! <<<")

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # history_data: [B, L, N, 4]
        
        enhanced_spa_emb = self.spatial_module.get_enhanced_spatial_embedding(self.spa_codebook)
        
        # 1. 主干输入
        main_input = history_data[..., :3]

        # 2. Patching
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat((main_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), main_input), dim=1)
        else:
            main_input_aug = main_input

        # 3. Encoders
        downsamp_input = [main_input_aug[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3) 

        patch_predict = self.patch_encoder(patch_input, spatial_codebook=enhanced_spa_emb)
        downsamp_predict = self.downsamp_encoder(downsamp_input, spatial_codebook=enhanced_spa_emb)

        # 4. Residual
        res_input = history_data[..., 0:1].permute(0, 1, 2, 3)
        res_out = self.residual(res_input)

        # 5. Base Output
        output = patch_predict + downsamp_predict + res_out

        # 6. Spatial Refinement
        history_flow = history_data[..., 0]
        output = self.spatial_module.refine_prediction(output, history_flow)
        
        # 🔥 关键点 4: 执行 MLP 分支
        if self.input_dim > 3:
            prior_data = history_data[..., 3:4] # [B, L, N, 1]
            
            # 这里调用的是 SimpleMLP
            kan_prior = self.prior_kan(prior_data)
            
            # 时间步对齐
            if self.input_len != self.output_len:
                kan_prior = kan_prior.permute(0, 2, 3, 1)
                kan_prior = self.time_proj(kan_prior)
                kan_prior = kan_prior.permute(0, 3, 1, 2)
            
            output = output + kan_prior

        return output