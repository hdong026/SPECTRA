import os
import sys
from easydict import EasyDict

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from basicts.losses import masked_mae
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners.runner_zoo.flow_matching_runner import FlowMatchingRunner
from basicts.archs.arch_zoo.KAFM.KAFM_arch import LatentSpectralKAFM

CFG = EasyDict()

# --- 通用 ---
CFG.DESCRIPTION = "Pure ST-KAFM PEMS04 (Final Fix)"
CFG.RUNNER = FlowMatchingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "PEMS04"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

# --- 环境 ---
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# --- 模型 ---
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "LatentSpectralKAFM"
CFG.MODEL.ARCH = LatentSpectralKAFM

CFG.MODEL.PARAM = {
    "cfgs": {
        "hidden_dim": 64,   
        "grid_size": 10,
        "top_k": None       
    },
    "kasa_backbone_args": {
        "node_size": 307,
        "input_len": CFG.DATASET_INPUT_LEN,
        "output_len": CFG.DATASET_OUTPUT_LEN,
        "adj_mx_path": os.path.join("datasets", CFG.DATASET_NAME, "adj_mx.pkl"),
        "input_dim": 4, 
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32, "d_dw": 32, "d_d": 32, "d_spa": 32,
        "if_time_in_day": True, "if_day_in_week": True, "if_spatial": True,
        "num_layer": 2, "spatial_scheme": "C",
        "use_gcn": True, "gcn_hidden_dim": 64,
        "use_dynamic_spatial": True, "dyn_hidden_dim": 64,
        "use_adaptive_adj": True, "adp_hidden_dim": 32,
        "use_hybrid_graph": True, "hybrid_alpha": 0.2
    }
}

CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0] 

# --- 训练 ---
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "AdamW" 
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,           
    "weight_decay": 1e-3,  
}
CFG.TRAIN.CLIP_GRAD = 5

CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 15, 50, 80], 
    "gamma": 0.5
}

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)]))
CFG.TRAIN.DATA = EasyDict()
# 关键：NULL_VAL 设为 0.0，因为归一化数据通常不包含 NaN，我们用 Mask 处理
CFG.TRAIN.NULL_VAL = 0.0 
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 32  
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# --- 验证/测试 ---
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False