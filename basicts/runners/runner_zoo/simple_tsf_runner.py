import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.
        Args:
            data: [B, L, N, C]
        Returns:
            data: [B, L, N, C_selected]
        """
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.
        Args:
            data: prediction with arbitrary shape, expected [B,L,N,C]
        Returns:
            data: [B,L,N,C_selected]
        """
        if self.target_features is not None:
            data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """
        Feed forward for train/val/test.

        data can be either:
          - (future_data, history_data)
          - (dataset_name, (future_data, history_data))   [from InterleavedLoader]
        Returns:
          prediction, real_value  (both [B,L,N,C])
        """

        active_dataset_name = None

        # ---- compat: accept (dataset_name, batch) ----
        if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], str):
            active_dataset_name = data[0]
            data = data[1]

        future_data, history_data = data

        # move to device
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)

        batch_size, length, num_nodes, _ = future_data.shape

        # feature selection
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # model forward
        if self.cl_param is None:
            prediction_data = self.model(
                history_data=history_data,
                future_data=future_data_4_dec,
                batch_seen=iter_num,
                epoch=epoch,
                train=train,
                active_dataset=active_dataset_name,  # optional for logging/diagnostics
            )
        else:
            task_level = self.curriculum_learning(epoch)
            prediction_data = self.model(
                history_data=history_data,
                future_data=future_data_4_dec,
                batch_seen=iter_num,
                epoch=epoch,
                train=train,
                task_level=task_level,
                active_dataset=active_dataset_name,
            )

        # output shape check
        assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, reshape it to [B, L, N, C]"

        # post-process: select target features
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)

        return prediction, real_value