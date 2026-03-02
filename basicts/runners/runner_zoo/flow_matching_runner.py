import torch
import os
import pickle
import sys
import numpy as np

# === 关键导入：确保 SCALER_REGISTRY 能找到 're_standard_transform' ===
from basicts.data.transform import *
# ===================================================================

try:
    from basicts.runners import BaseTimeSeriesForecastingRunner
except ImportError:
    try:
        from basicts.runners.base_ts_runner import BaseTimeSeriesForecastingRunner
    except ImportError:
        from basicts.runners.runner_zoo.simple_ts_runner import SimpleTimeSeriesForecastingRunner as BaseTimeSeriesForecastingRunner

class FlowMatchingRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.clip_grad = cfg.get("TRAIN", {}).get("CLIP_GRAD", None)

        # =================================================================
        # FIX 1: 正确加载 Scaler，保留完整结构
        # =================================================================
        scaler_filename = f"scaler_in{cfg.get('DATASET_INPUT_LEN', 12)}_out{cfg.get('DATASET_OUTPUT_LEN', 12)}.pkl"
        scaler_path = os.path.join(cfg.TRAIN.DATA.DIR, scaler_filename)
        
        if not os.path.exists(scaler_path):
             scaler_path = os.path.join(cfg.TRAIN.DATA.DIR, "scaler.pkl")

        if os.path.exists(scaler_path):
            self.logger.info(f"Loading scaler explicitly from {scaler_path}")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                # 关键：不要去动它的结构！保持 {'func': ..., 'args': ...}
                # 这样 BaseTimeSeriesForecastingRunner 才能自动处理反归一化
                self.logger.info(f"Scaler loaded successfully. Keys: {self.scaler.keys() if isinstance(self.scaler, dict) else 'Not a dict'}")
        else:
            self.logger.warning(f"CRITICAL WARNING: Scaler file not found at {scaler_path}! Metrics will be WRONG.")
        # =================================================================

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :, :, self.target_features]

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        future_data, history_data = data
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        
        batch_size, length, num_nodes, _ = future_data.shape
        
        # 提取输入特征
        history_input = self.select_input_features(history_data)
        
        # === 终极排查：通道指纹识别 (Channel Fingerprinting) ===
        # 只在第一轮打印，看看模型到底吃进了什么
        if not getattr(self, '_channel_debug_printed', False):
            self._channel_debug_printed = True
            print("\n" + "#"*60)
            print("CHANNEL FINGERPRINTING (Input to Model)")
            print(f"History Input Shape: {history_input.shape}") # [B, L, N, C]
            
            C = history_input.shape[-1]
            for c in range(C):
                # 取第 c 个通道的数据
                ch_data = history_input[..., c]
                mean_val = ch_data.mean().item()
                std_val = ch_data.std().item()
                min_val = ch_data.min().item()
                max_val = ch_data.max().item()
                
                print(f"Channel {c}: Mean={mean_val:.4f} | Std={std_val:.4f} | Range=[{min_val:.2f}, {max_val:.2f}]")
                
                # 智能推断
                guess = "Unknown"
                if abs(mean_val) < 0.2 and 0.5 < std_val < 1.5: guess = "Flow/Prior (Normalized)"
                elif 0.2 < mean_val < 0.8 and std_val < 0.5: guess = "TOD (Time)"
                elif mean_val > 2.0: guess = "DOW (Week)"
                
                print(f"  -> Likely: {guess}")

            print("#"*60 + "\n")
        # ==========================================================

        if train:
            # === 训练模式 ===
            fm_loss = self.model(
                history_data=history_input, 
                future_data=future_data, 
                batch_seen=iter_num, epoch=epoch, train=True
            )
            prediction = fm_loss.view(1).expand(batch_size, length, num_nodes, 1)
            target = torch.zeros_like(prediction)
            return prediction, target
            
        else:
            # === 推理模式 ===
            prediction = self.model(
                history_data=history_input, 
                future_data=None, 
                batch_seen=iter_num, epoch=epoch, train=False
            )
            
            real_value = self.select_target_features(future_data)
            
            if prediction.shape[1] != length:
                 prediction = prediction[:, :length, :, :]

            return prediction, real_value

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()