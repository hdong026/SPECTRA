# examples/cfg_tkde.py  (UPDATED: add USE_CANON switch for ST-GFM Scheme-B)

import os
import sys
from easydict import EasyDict

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from basicts.losses import masked_mae
from basicts.data.dataset import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner

# ---- original model ----
from basicts.archs import KASA_TKDE

# ---- new canonical dataset + model ----
from basicts.data.canon_dataset import CanonicalHistoryDataset
from basicts.archs.arch_zoo.KASA_arch_v2.KASA_CANON_GFM import KASA_CANON_GFM

CFG = EasyDict()

# ===== protocol overrides (env-driven) =====
CFG.DATASET_NAME = os.getenv("DATASET_NAME", "PEMS04")
CFG.RUN_TAG = os.getenv("RUN_TAG", "")
CFG.MODE = os.getenv("MODE", "train").lower()                 # train / finetune / test
CFG.FEWSHOT_RATIO = float(os.getenv("FEWSHOT_RATIO", "1.0"))
CFG.RESUME_FROM = os.getenv("RESUME_FROM", "")

# ===== new switch: canonical ST-GFM mode =====
CFG.USE_CANON = os.getenv("USE_CANON", "0").lower() in ["1", "true", "yes", "y"]

# [NEW] canonical interface dims (m,k)
CFG.CANON_M = int(os.getenv("CANON_M", "12"))        # temporal basis dim m
CFG.TOKEN_K = int(os.getenv("TOKEN_K", "64"))        # graph basis dim k (U_k)
CFG.CANON_BASIS = os.getenv("CANON_BASIS", "dct")    # time basis
CFG.USE_CALIB = os.getenv("USE_CALIB", "1").lower() in ["1", "true", "yes", "y"]

# [NEW] common runtime overrides
CFG.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
CFG.TEST_INTERVAL = int(os.getenv("TEST_INTERVAL", "1"))
CFG.NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
CFG.SEED = int(os.getenv("SEED", "1"))

# [NEW] scaling preset: S/M/L
# S: d=32,l=2 ; M: d=64,l=3 ; L: d=128,l=3
CFG.SCALE = os.getenv("SCALE", "S").upper()
SCALE_MAP = {
    "S": dict(d=32, layers=2, gcn=64, dyn=64, adp=32),
    "M": dict(d=64, layers=3, gcn=128, dyn=128, adp=64),
    "L": dict(d=128, layers=3, gcn=256, dyn=256, adp=128),
}
scale_cfg = SCALE_MAP.get(CFG.SCALE, SCALE_MAP["S"])

NODE_SIZE = {"PEMS04": 307, "PEMS07": 883, "PEMS08": 170, "PEMS-BAY": 325}
CFG.NODE_SIZE = NODE_SIZE[CFG.DATASET_NAME]

# ================= general ================= #
CFG.DESCRIPTION = "TKDE ST-GFM configuration (KASA backbone + canonical interface)"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

# default dataset class (may be overridden below)
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = CFG.SEED
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()

# --------- Mode A: original node-domain KASA_TKDE (single-graph / baseline) ----------
if not CFG.USE_CANON:
    CFG.MODEL.NAME = "KASA_TKDE"
    CFG.MODEL.ARCH = KASA_TKDE

    # optional: whether to use prior channel (your old setup)
    use_prior = os.getenv("USE_PRIOR", "1").lower() in ["1", "true", "yes", "y"]
    # input_dim is number of forward features: flow + tod + dow (+ prior)
    input_dim = 4 if use_prior else 3

    d = scale_cfg["d"]
    CFG.MODEL.PARAM = {
        "node_size": CFG.NODE_SIZE,
        "input_len": CFG.DATASET_INPUT_LEN,
        "output_len": CFG.DATASET_OUTPUT_LEN,
        "input_dim": input_dim,

        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,

        # scale-aware dims
        "d_td": d,
        "d_dw": d,
        "d_d": d,
        "d_spa": d,
        "num_layer": scale_cfg["layers"],

        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,

        "spatial_scheme": "C",
        "adj_mx_path": os.path.join("datasets", CFG.DATASET_NAME, "adj_mx.pkl"),

        "use_gcn": True,
        "gcn_hidden_dim": scale_cfg["gcn"],

        "use_dynamic_spatial": True,
        "dyn_hidden_dim": scale_cfg["dyn"],
        "dyn_topk": int(os.getenv("DYN_TOPK", "20")),
        "dyn_tau": float(os.getenv("DYN_TAU", "0.5")),
        "dyn_static_weight": float(os.getenv("DYN_STATIC_WEIGHT", "0.2")),

        "use_adaptive_adj": True,
        "adp_hidden_dim": scale_cfg["adp"],
        "adp_topk": int(os.getenv("ADP_TOPK", "20")),
        "adp_tau": float(os.getenv("ADP_TAU", "0.5")),

        "use_hybrid_graph": True,
        "hybrid_alpha": float(os.getenv("HYBRID_ALPHA", "0.2")),

        "use_lightweight_spatial": False,

        # keep token_k for compatibility if you still want it
        "token_k": CFG.TOKEN_K,
    }

    # forward/target features
    if use_prior:
        CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
    else:
        CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
    CFG.MODEL.TARGET_FEATURES = [0]


# --------- Mode B: canonical ST-GFM (Scheme-B reference calibration) ----------
else:
    CFG.MODEL.NAME = "KASA_CANON_GFM"
    CFG.MODEL.ARCH = KASA_CANON_GFM

    # In canonical mode, history length becomes m (CANON_M), node-size becomes k (TOKEN_K).
    # Output is still node-domain [B,H,N,1] after multiplying U_k^T inside the model.
    d = scale_cfg["d"]
    CFG.MODEL.PARAM = {
        # keep original node_size for bookkeeping (output N), but model does not depend on it
        "node_size": CFG.NODE_SIZE,

        # canonical sequence length m and forecast length H
        "input_len": CFG.CANON_M,
        "output_len": CFG.DATASET_OUTPUT_LEN,

        # KASA-style multi-scale params
        "patch_len": int(os.getenv("PATCH_LEN", "3")),
        "stride": int(os.getenv("STRIDE", "4")),

        # backbone width/depth (reuse scale preset)
        "d_d": d,
        "num_layer": scale_cfg["layers"],

        # canonical token dim k (graph basis)
        "token_k": CFG.TOKEN_K,
    }

    # Switch dataset class to canonical dataset
    CFG.DATASET_CLS = CanonicalHistoryDataset
    CFG.DATASET_ARGS = {
        "dataset_name": CFG.DATASET_NAME,   # helps canonical dataset locate Uk/ab cache
        "T": CFG.DATASET_INPUT_LEN,         # original history window length for slicing
        "H": CFG.DATASET_OUTPUT_LEN,
        "m": CFG.CANON_M,
        "k": CFG.TOKEN_K,
        "basis": CFG.CANON_BASIS,
        "flow_channel": 0,
        "cache_root": "datasets",
        "use_calib": CFG.USE_CALIB,
    }

    # In canonical dataset, history_data is [m,k,1], so only one input feature channel.
    CFG.MODEL.FORWARD_FEATURES = [0]
    CFG.MODEL.TARGET_FEATURES = [0]


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": float(os.getenv("LR", "0.002")),
    "weight_decay": float(os.getenv("WD", "0.0001")),
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {"milestones": [1, 35, 60, 80, 95], "gamma": 0.5}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = CFG.NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    f"{CFG.MODEL.NAME}_{CFG.DATASET_NAME}_{CFG.RUN_TAG}_{CFG.TRAIN.NUM_EPOCHS}"
)

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = CFG.NUM_WORKERS
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = CFG.NUM_WORKERS
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = CFG.TEST_INTERVAL
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = CFG.NUM_WORKERS
CFG.TEST.DATA.PIN_MEMORY = False