import math
import functools
from typing import Tuple, Union, Optional

import os
import json
import time
import inspect

import torch
import numpy as np
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import masked_mae, masked_mape, masked_rmse

import pickle
import scipy.sparse as sp

from torch.utils.data import DataLoader
from ..data.interleaved_loader import InterleavedLoader


# ============================================================
# Compat wrapper: EasyTorch expects train_data_loader.sampler
# when calling on_epoch_start(). InterleavedLoader may not have it.
# ============================================================
class _LoaderSamplerCompat:
    """
    Wrap any iterable loader to provide a `.sampler` attribute expected by EasyTorch.
    """
    def __init__(self, loader):
        self._loader = loader
        self.sampler = None  # EasyTorch only reads it; None is fine

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        # fallback if underlying loader doesn't implement __len__
        try:
            return len(self._loader)
        except Exception:
            return 0

    def __getattr__(self, name):
        # delegate other attributes (e.g., iters_per_epoch, probs, etc.)
        return getattr(self._loader, name)


# ============================================================
# PEFT helper: freeze all, then unfreeze a small allowlist
# ============================================================
def apply_peft_freeze(model, logger=None):
    for p in model.parameters():
        p.requires_grad = False

    allow = [
        # topology-agnostic shared coeffs
        "Z_spa",
        "Z_src",
        "Z_dst",
        # Phase-III spectral params
        "spectral_gate",
        "freq_modulator",
        # generic heads/projections
        "head",
        "output",
        "proj",
        # optional adapters / norms / biases
        "adapter",
        "lora",
        "bias",
        "norm",
        "ln",
    ]
    for n, p in model.named_parameters():
        if any(a in n for a in allow):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if logger:
        logger.info(f"[PEFT] trainable={trainable}/{total} ({100*trainable/total:.2f}%)")

    if trainable == 0:
        if logger:
            logger.info("[PEFT] allowlist matched 0 params; fallback to full finetune.")
        for p in model.parameters():
            p.requires_grad = True


# ============================================================
# Graph state (U_k) helpers: load/cache or cold-start compute
# ============================================================
def _load_adj_from_pkl(adj_mx_path: str):
    with open(adj_mx_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")

    if isinstance(obj, (list, tuple)):
        adj = obj[2]
    elif isinstance(obj, dict):
        adj = obj.get("adj_mx", None)
    else:
        adj = obj
    if adj is None:
        raise ValueError("adj_mx not found in adj_mx.pkl")
    if sp.issparse(adj):
        adj = adj.toarray()
    adj = np.array(adj, dtype=np.float32)
    adj = np.maximum(adj, adj.T)
    return adj


def _laplacian_topk_eigvecs_and_vals(adj: np.ndarray, k: int):
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    An = D_inv_sqrt @ A @ D_inv_sqrt
    L = np.eye(A.shape[0], dtype=np.float32) - An

    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    for i in range(vecs.shape[1]):
        j = np.argmax(np.abs(vecs[:, i]))
        if vecs[j, i] < 0:
            vecs[:, i] *= -1.0

    k = min(k, vecs.shape[1])
    return vecs[:, :k].astype(np.float32), vals[:k].astype(np.float32)


def load_or_compute_graph_state(dataset_name: str, token_k: int, logger=None):
    ddir = os.path.join("datasets", dataset_name)
    uk_path = os.path.join(ddir, "graph_state_Uk.npy")
    ev_path = os.path.join(ddir, "graph_state_eigvals.npy")

    if os.path.exists(uk_path):
        Uk = np.load(uk_path).astype(np.float32)
        Uk = Uk[:, :token_k] if Uk.shape[1] >= token_k else Uk
        eig = None
        if os.path.exists(ev_path):
            eig = np.load(ev_path).astype(np.float32)
            eig = eig[:token_k] if eig.shape[0] >= token_k else eig
        if logger:
            logger.info(f"[GRAPH_STATE] loaded cached U_k: {uk_path} (N={Uk.shape[0]}, k={Uk.shape[1]})")
        return Uk, eig

    adj_mx_path = os.path.join(ddir, "adj_mx.pkl")
    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(f"adj_mx.pkl not found for dataset {dataset_name}: {adj_mx_path}")

    adj = _load_adj_from_pkl(adj_mx_path)
    Uk, eig = _laplacian_topk_eigvecs_and_vals(adj, token_k)

    os.makedirs(ddir, exist_ok=True)
    np.save(uk_path, Uk)
    np.save(ev_path, eig)
    if logger:
        logger.info(f"[GRAPH_STATE] cold-start computed + cached: {uk_path} (N={Uk.shape[0]}, k={Uk.shape[1]})")
    return Uk, eig


def _call_set_graph_state(model: torch.nn.Module, Uk: torch.Tensor, eig: Optional[torch.Tensor]):
    """
    Compatibility: support set_graph_state(Uk, eig) or set_graph_state(Uk).
    """
    fn = getattr(model, "set_graph_state", None)
    if fn is None or not callable(fn):
        return False
    try:
        fn(Uk, eig)  # preferred
        return True
    except TypeError:
        fn(Uk)       # fallback
        return True


def attach_graph_state_to_model(model: torch.nn.Module,
                                dataset_name: str,
                                token_k: int,
                                to_running_device,
                                logger=None):
    Uk_np, eig_np = load_or_compute_graph_state(dataset_name, token_k, logger=logger)

    Uk = torch.from_numpy(Uk_np).float()
    Uk = to_running_device(Uk)

    eig = None
    if eig_np is not None:
        eig = torch.from_numpy(eig_np).float()
        eig = to_running_device(eig)

    if hasattr(model, "set_graph_state") and callable(getattr(model, "set_graph_state")):
        _call_set_graph_state(model, Uk, eig)
    else:
        model.register_buffer("U_k", Uk, persistent=False)
        if eig is not None:
            model.register_buffer("eigvals_k", eig, persistent=False)

    if logger:
        logger.info(f"[GRAPH_STATE] attached to model for {dataset_name}: U_k={tuple(Uk.shape)}")


# ============================================================
# Cross-graph checkpoint loading: shape-compatible only
# ============================================================
def load_state_dict_compatible(model: torch.nn.Module, state_dict: dict, logger=None):
    model_sd = model.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if logger is not None:
        logger.info(
            f"[CKPT] compatible load: loaded={len(filtered)}, skipped={len(skipped)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        logger.info(f"[CKPT] example skipped keys: {skipped[:10]}")
    return len(filtered), skipped, missing, unexpected


# ============================================================
# Runner
# ============================================================
class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for short term multivariate time series forecasting datasets.
    Supports:
      - interleaved multi-graph pretraining
      - graph-state attachment per dataset
      - train_on_normalized for mixed pretrain
      - optional PEFT
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.dataset_name = cfg["DATASET_NAME"]
        self.run_mode = os.getenv("MODE", "train").lower()
        self.resume_from = os.getenv("RESUME_FROM", "")
        self.run_tag = os.getenv("RUN_TAG", "")
        self.save_pred = os.getenv("SAVE_PRED", "0").lower() in ["1", "true", "yes", "y"]
        self.pred_save_dir = os.getenv("PRED_SAVE_DIR", "predictions")
        self.sign_flip_trials = int(os.getenv("SIGN_FLIP_TRIALS", "0"))

        self.token_k = int(cfg.get("TOKEN_K", int(os.getenv("TOKEN_K", "64"))))

        self.null_val = cfg["TRAIN"].get("NULL_VAL", np.nan)
        self.dataset_type = cfg["DATASET_TYPE"]
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)

        self.scaler = load_pkl(
            "{0}/scaler_in{1}_out{2}.pkl".format(
                cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
            )
        )

        self.loss = cfg["TRAIN"]["LOSS"]
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}

        self.cl_param = cfg.TRAIN.get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg.TRAIN.CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg.TRAIN.CL.get("CL_EPOCHS")
            self.prediction_length = cfg.TRAIN.CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg.TRAIN.CL.get("STEP_SIZE", 1)

        self.evaluation_horizons = [_ - 1 for _ in cfg["TEST"].get("EVALUATION_HORIZONS", range(1, 13))]
        assert min(self.evaluation_horizons) >= 0

        # state for mixed-pretrain
        self._mix_states = None
        self._val_graph_state_set = False

    # -----------------------------
    def init_training(self, cfg: dict):
        super().init_training(cfg)

        # ---- interleaved multi-graph pretraining ----
        multi = os.getenv("MULTI_DATASETS", "").strip()
        if self.run_mode == "train" and multi:
            ds_list = [x.strip() for x in multi.split(",") if x.strip()]
            assert len(ds_list) >= 2
            self.logger.info(f"[MIX] enable interleaved pretraining over: {ds_list}")

            # preload graph states + normalized adjacency per dataset
            self._mix_states = {}
            for name in ds_list:
                Uk_np, eig_np = load_or_compute_graph_state(name, self.token_k, logger=self.logger)
                Uk = self.to_running_device(torch.from_numpy(Uk_np).float())
                eig = self.to_running_device(torch.from_numpy(eig_np).float()) if eig_np is not None else None

                adj_path = os.path.join("datasets", name, "adj_mx.pkl")
                adj = _load_adj_from_pkl(adj_path)
                A = adj + np.eye(adj.shape[0], dtype=np.float32)
                d = A.sum(1)
                d_inv_sqrt = np.power(d, -0.5)
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
                D = np.diag(d_inv_sqrt.astype(np.float32))
                An = (D @ A @ D).astype(np.float32)
                An_t = self.to_running_device(torch.from_numpy(An).float())

                self._mix_states[name] = {"U_k": Uk, "eig": eig, "adj_norm": An_t}

            # per-dataset dataloaders
            bs = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
            nw = cfg["TRAIN"]["DATA"]["NUM_WORKERS"]
            pin = cfg["TRAIN"]["DATA"].get("PIN_MEMORY", False)

            loaders = {}
            total_batches = 0
            for name in ds_list:
                ddir = os.path.join("datasets", name)
                data_file_path = f"{ddir}/data_in{cfg['DATASET_INPUT_LEN']}_out{cfg['DATASET_OUTPUT_LEN']}.pkl"
                index_file_path = f"{ddir}/index_in{cfg['DATASET_INPUT_LEN']}_out{cfg['DATASET_OUTPUT_LEN']}.pkl"
                dataset = cfg["DATASET_CLS"](data_file_path=data_file_path, index_file_path=index_file_path, mode="train")
                dl = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True)
                loaders[name] = dl
                total_batches += max(1, len(dl))

            mix_mode = os.getenv("MIX_PROBS", "size").lower()
            if mix_mode == "uniform":
                probs = {k: 1.0 for k in ds_list}
            else:
                probs = {k: float(len(loaders[k])) for k in ds_list}

            iters = int(os.getenv("MIX_ITERS_PER_EPOCH", str(total_batches)))
            self.iter_per_epoch = iters

            inter = InterleavedLoader(loaders, iters_per_epoch=iters, probs=probs, seed=int(cfg["ENV"]["SEED"]))
            # ✅ fix EasyTorch sampler access
            self.train_data_loader = _LoaderSamplerCompat(inter)

            self.logger.info(f"[MIX] iters_per_epoch={iters}, probs={probs}")

            # during mixed pretrain, compute loss on normalized scale
            self.train_on_normalized = True
        else:
            self.train_on_normalized = False

        # ---- attach graph state for the "default" dataset before any forward ----
        attach_graph_state_to_model(self.model, self.dataset_name, self.token_k, self.to_running_device, logger=self.logger)

        # ---- warm-start ----
        if self.run_mode in ["train", "finetune"] and self.resume_from:
            ckpt = torch.load(self.resume_from, map_location="cpu")
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
            load_state_dict_compatible(self.model, state, logger=self.logger)
            attach_graph_state_to_model(self.model, self.dataset_name, self.token_k, self.to_running_device, logger=self.logger)

        # ---- PEFT ----
        self.enable_peft = os.getenv("PEFT", "0").lower() in ["1", "true", "yes", "y"]
        if self.run_mode == "finetune" and self.enable_peft:
            apply_peft_freeze(self.model, logger=self.logger)
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(
                params,
                lr=cfg["TRAIN"]["OPTIM"]["PARAM"]["lr"],
                weight_decay=cfg["TRAIN"]["OPTIM"]["PARAM"].get("weight_decay", 0.0),
            )

        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_" + key, "train", "{:.4f}")

    def _set_active_graph(self, name: str):
        """
        Switch graph state (U_k / eig) and (optionally) static adj to the given dataset name.
        This must be called BEFORE forwarding a batch from that dataset.
        """
        if self._mix_states is not None and name in self._mix_states:
            st = self._mix_states[name]
            if hasattr(self.model, "set_graph_state"):
                _call_set_graph_state(self.model, st["U_k"], st.get("eig", None))
            else:
                # fallback buffers
                self.model.register_buffer("U_k", st["U_k"], persistent=False)
                if st.get("eig", None) is not None:
                    self.model.register_buffer("eigvals_k", st["eig"], persistent=False)

            # optional: update static adjacency if your spatial_module supports it
            if hasattr(self.model, "spatial_module") and hasattr(self.model.spatial_module, "set_static_adj"):
                self.model.spatial_module.set_static_adj(st["adj_norm"])
            return

        attach_graph_state_to_model(self.model, name, self.token_k, self.to_running_device, logger=self.logger)

    # -----------------------------
    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_" + key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_" + key, "test", "{:.4f}")

    # -----------------------------
    def build_train_dataset(self, cfg: dict):
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(
            cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )

        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(
            cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(
            cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )

        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(
            cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]
        )

        dataset_args = cfg.get("DATASET_ARGS", {})
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))
        return dataset

    # -----------------------------
    def curriculum_learning(self, epoch: int = None) -> int:
        if epoch is None:
            return self.prediction_length
        epoch -= 1
        if epoch < self.warm_up_epochs:
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            metric_item = metric_func(*args)
        elif callable(metric_func):
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

    # -----------------------------
    def _maybe_switch_by_batch(self, data):
        """
        If InterleavedLoader yields (dataset_name, batch), switch graph state here.
        Returns the pure batch that downstream forward() expects.
        """
        if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], str):
            name, batch = data[0], data[1]
            self._set_active_graph(name)
            return batch
        return data

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index

        # ✅ critical: per-batch graph state switch for mixed pretrain
        data = self._maybe_switch_by_batch(data)

        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        pred = forward_return[0]
        real = forward_return[1]
        aux_loss = forward_return[2] if len(forward_return) > 2 else 0.0

        if getattr(self, "train_on_normalized", False):
            if self.cl_param:
                cl_length = self.curriculum_learning(epoch=epoch)
                pred_for_loss = pred[:, :cl_length, :, :]
                real_for_loss = real[:, :cl_length, :, :]
            else:
                pred_for_loss = pred
                real_for_loss = real

            forecast_loss = self.metric_forward(self.loss, [pred_for_loss, real_for_loss])

            if torch.is_tensor(aux_loss):
                aux_term = aux_loss.to(device=forecast_loss.device, dtype=forecast_loss.dtype)
            else:
                aux_term = torch.tensor(float(aux_loss), device=forecast_loss.device, dtype=forecast_loss.dtype)

            loss = forecast_loss + aux_term

            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                self.update_epoch_meter("train_" + metric_name, metric_item.item())

            return loss

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(pred, **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(real, **self.scaler["args"])

        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            pred_for_loss = prediction_rescaled[:, :cl_length, :, :]
            real_for_loss = real_value_rescaled[:, :cl_length, :, :]
        else:
            pred_for_loss = prediction_rescaled
            real_for_loss = real_value_rescaled

        forecast_loss = self.metric_forward(self.loss, [pred_for_loss, real_for_loss])

        if torch.is_tensor(aux_loss):
            aux_term = aux_loss.to(device=forecast_loss.device, dtype=forecast_loss.dtype)
        else:
            aux_term = torch.tensor(float(aux_loss), device=forecast_loss.device, dtype=forecast_loss.dtype)

        loss = forecast_loss + aux_term

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("train_" + metric_name, metric_item.item())

        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        # ✅ ensure val runs on the configured dataset graph state (not last mixed batch)
        if self._mix_states is not None and not self._val_graph_state_set:
            self._set_active_graph(self.dataset_name)
            self._val_graph_state_set = True

        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_" + metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self):
        if self.run_mode == "test":
            attach_graph_state_to_model(self.model, self.dataset_name, self.token_k, self.to_running_device, logger=self.logger)

            if self.resume_from:
                ckpt = torch.load(self.resume_from, map_location="cpu")
                state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
                load_state_dict_compatible(self.model, state, logger=self.logger)
                attach_graph_state_to_model(self.model, self.dataset_name, self.token_k, self.to_running_device, logger=self.logger)
                self.logger.info(f"[CKPT] Loaded for test: {self.resume_from}")

        if self._mix_states is not None:
            self._set_active_graph(self.dataset_name)

        prediction = []
        real_value = []
        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0])
            real_value.append(forward_return[1])

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)

        prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])

        if self.save_pred:
            os.makedirs(self.pred_save_dir, exist_ok=True)
            meta = {
                "dataset": self.dataset_name,
                "run_tag": self.run_tag,
                "mode": self.run_mode,
                "resume_from": self.resume_from,
                "fewshot_ratio": float(os.getenv("FEWSHOT_RATIO", "1.0")),
                "token_k": int(self.token_k),
                "timestamp": time.time(),
                "shape": list(prediction.shape),
                "scaler_func": self.scaler.get("func", ""),
            }
            save_path = os.path.join(self.pred_save_dir, f"{self.dataset_name}_{self.run_tag}.npz")
            np.savez_compressed(
                save_path,
                y_pred=prediction.detach().cpu().numpy(),
                y_true=real_value.detach().cpu().numpy(),
                meta=json.dumps(meta),
            )
            self.logger.info(f"[SAVE_PRED] {save_path}")

        for i in self.evaluation_horizons:
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]
            metric_repr = ""
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_repr += ", Test {0}: {1:.4f}".format(metric_name, metric_item.item())
            log = "Evaluate best model on test data for horizon {:d}" + metric_repr
            self.logger.info(log.format(i + 1))

        for metric_name, metric_func in self.metrics.items():
            if self.evaluate_on_gpu:
                metric_item = self.metric_forward(metric_func, [prediction, real_value])
            else:
                metric_item = self.metric_forward(metric_func, [prediction.detach().cpu(), real_value.detach().cpu()])
            self.update_epoch_meter("test_" + metric_name, metric_item.item())

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)
        self._val_graph_state_set = False