import os

import torch
from torch.utils.data import Dataset

from ..utils import load_pkl


class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)

        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()

        self.index = load_pkl(index_file_path)[mode]

        # ---- few-shot support (env-driven, minimal) ----
        env_mode = os.getenv("MODE", "train").lower()         # train / finetune / test
        fewshot_ratio = float(os.getenv("FEWSHOT_RATIO", "1.0"))
        if mode == "train" and env_mode == "finetune" and fewshot_ratio < 1.0:
            full_len = len(self.index)
            keep = max(1, int(full_len * fewshot_ratio))
            self.index = self.index[:keep]

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"BasicTS can not find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"BasicTS can not find index file {index_file_path}")

    def __getitem__(self, index: int) -> tuple:
        idx = list(self.index[index])
        if isinstance(idx[0], int):
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
        else:
            # discontinuous index or custom index
            history_index = list(idx[0])  # IMPORTANT: copy to avoid in-place growth bug
            assert idx[1] not in history_index, "current time t should not included in idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
        return future_data, history_data

    def __len__(self):
        return len(self.index)


class TimeSeriesForecastingDataset_ZhengZhou(Dataset):
    """Time series forecasting dataset with tpl matrix."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)

        data = load_pkl(data_file_path)
        self.data = torch.from_numpy(data["processed_data"]).float()
        self.tpl_matrix = torch.from_numpy(data["processed_tpl"]).float()
        self.index = load_pkl(index_file_path)[mode]

        # ---- few-shot support (same policy) ----
        env_mode = os.getenv("MODE", "train").lower()
        fewshot_ratio = float(os.getenv("FEWSHOT_RATIO", "1.0"))
        if mode == "train" and env_mode == "finetune" and fewshot_ratio < 1.0:
            full_len = len(self.index)
            keep = max(1, int(full_len * fewshot_ratio))
            self.index = self.index[:keep]

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"BasicTS can not find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"BasicTS can not find index file {index_file_path}")

    def __getitem__(self, index: int) -> tuple:
        idx = list(self.index[index])
        if isinstance(idx[0], int):
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]
            history_tpl = self.tpl_matrix[idx[0]:idx[1]]
        else:
            history_index = list(idx[0])  # copy!
            assert idx[1] not in history_index, "current time t should not included in idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
            history_tpl = self.tpl_matrix[history_index]
        return future_data, history_data, history_tpl

    def __len__(self):
        return len(self.index)