import os
import sys
import uuid
import importlib.util
import argparse

def load_cfg(cfg_path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFG

def ensure_md5(cfg):
    if "MD5" not in cfg or not cfg["MD5"]:
        cfg["MD5"] = os.getenv("MD5", "") or uuid.uuid4().hex
    return cfg

def add_dotkey_aliases(cfg):
    # MODEL
    if "MODEL" in cfg:
        cfg["MODEL.NAME"] = cfg["MODEL"].get("NAME", cfg.get("MODEL.NAME", None))
        if "PARAM" in cfg["MODEL"]:
            cfg["MODEL.PARAM"] = cfg["MODEL"]["PARAM"]

    # TRAIN
    if "TRAIN" in cfg:
        cfg["TRAIN.NUM_EPOCHS"] = cfg["TRAIN"].get("NUM_EPOCHS", cfg.get("TRAIN.NUM_EPOCHS", None))
        cfg["TRAIN.CKPT_SAVE_DIR"] = cfg["TRAIN"].get("CKPT_SAVE_DIR", cfg.get("TRAIN.CKPT_SAVE_DIR", None))
        if "DATA" in cfg["TRAIN"]:
            cfg["TRAIN.DATA"] = cfg["TRAIN"]["DATA"]

        if "OPTIM" in cfg["TRAIN"]:
            cfg["TRAIN.OPTIM"] = cfg["TRAIN"]["OPTIM"]
        if "LR_SCHEDULER" in cfg["TRAIN"]:
            cfg["TRAIN.LR_SCHEDULER"] = cfg["TRAIN"]["LR_SCHEDULER"]

    # VAL / TEST
    if "VAL" in cfg:
        cfg["VAL.INTERVAL"] = cfg["VAL"].get("INTERVAL", cfg.get("VAL.INTERVAL", 1))
        if "DATA" in cfg["VAL"]:
            cfg["VAL.DATA"] = cfg["VAL"]["DATA"]

    if "TEST" in cfg:
        cfg["TEST.INTERVAL"] = cfg["TEST"].get("INTERVAL", cfg.get("TEST.INTERVAL", 1))
        if "DATA" in cfg["TEST"]:
            cfg["TEST.DATA"] = cfg["TEST"]["DATA"]

    # ENV
    if "ENV" in cfg and "SEED" in cfg["ENV"]:
        cfg["ENV.SEED"] = cfg["ENV"]["SEED"]

    return cfg

if __name__ == "__main__":
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    cfg = add_dotkey_aliases(cfg)
    cfg = ensure_md5(cfg)

    # ===== key fix: add .has() for EasyTorch =====
    if not hasattr(cfg, "has"):
        cfg.has = lambda k: (k in cfg)

    runner = cfg.RUNNER(cfg)
    runner.test_process(cfg=cfg, train_epoch=None)