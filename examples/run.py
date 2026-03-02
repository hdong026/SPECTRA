import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
from basicts import launch_training

# NEW: direct runner creation for eval-only
from easytorch.config.utils import import_config
from easytorch.config.utils import init_cfg
from easytorch.core.runner import Runner  # only for type hints; not mandatory


torch.set_num_threads(1)  # avoid high cpu avg usage


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    parser.add_argument("-c", "--cfg", default="examples/LSTNN/LSTNN_PEMS04.py", help="training config")
    parser.add_argument("--gpus", default="1", help="visible gpus")
    # NEW: eval-only switch
    parser.add_argument("--eval_only", action="store_true", help="Only run test() once, no training.")
    return parser.parse_args()


def _run_eval_only(cfg_path: str, gpus: str):
    """
    Eval-only path:
      - load cfg
      - build runner
      - rely on runner.test() to load RESUME_FROM (your base_tsf_runner already supports it)
    """
    # Set visible GPUs like launch_training does
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

    # Import + init cfg (easytorch utilities)
    # init_cfg also expands ENV/paths similarly to training.
    cfg = import_config(cfg_path, verbose=True)
    cfg = init_cfg(cfg, save=False)

    # Build runner (same as training would)
    runner_cls = cfg["RUNNER"]
    runner = runner_cls(cfg)

    # IMPORTANT: runner.test() is decorated by master_only; in single GPU it's fine
    runner.init_test(cfg)
    runner.build_test_data_loader(cfg)

    # If your runner needs model on device etc, BaseRunner usually handles in init.
    # Here we simply call test().
    runner.test()


if __name__ == "__main__":
    args = parse_args()

    mode = os.getenv("MODE", "train").lower()
    # If MODE=test OR user passed --eval_only, go eval-only
    if mode == "test" or args.eval_only:
        _run_eval_only(args.cfg, args.gpus)
    else:
        launch_training(args.cfg, args.gpus)