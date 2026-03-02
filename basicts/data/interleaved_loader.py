# basicts/data/interleaved_loader.py
from __future__ import annotations

import random
from typing import Dict, Iterator, Optional, Tuple, Any

import numpy as np


class InterleavedLoader:
    """
    Interleave multiple PyTorch DataLoaders and yield batches from them.

    Key features:
      - yields (dataset_name, batch) so the runner can switch graph state per batch
      - has __len__ and .sampler to be compatible with frameworks that assume DataLoader-like API
      - supports weighted sampling across datasets (probs)
      - supports fixed iters_per_epoch (common for mixed pretraining)

    Args:
        loaders: dict[name -> dataloader]
        iters_per_epoch: number of batches to yield per epoch
        probs: sampling weights (unnormalized ok). if None -> uniform
        seed: RNG seed for reproducibility
        drop_last: if True, when a loader is exhausted, restart iterator (still yields exactly iters_per_epoch)
    """

    def __init__(
        self,
        loaders: Dict[str, Any],
        iters_per_epoch: int,
        probs: Optional[Dict[str, float]] = None,
        seed: int = 1,
        drop_last: bool = True,
    ):
        assert isinstance(loaders, dict) and len(loaders) > 0, "loaders must be non-empty dict"
        self.loaders = loaders
        self.names = list(loaders.keys())
        self.iters_per_epoch = int(iters_per_epoch)
        assert self.iters_per_epoch > 0

        # DataLoader-like compatibility:
        # EasyTorch might read train_data_loader.sampler
        self.sampler = None

        # Build sampling distribution
        if probs is None:
            self.probs = {k: 1.0 for k in self.names}
        else:
            # keep only keys that exist, fallback to 1.0
            self.probs = {k: float(probs.get(k, 1.0)) for k in self.names}

        w = np.array([self.probs[k] for k in self.names], dtype=np.float64)
        w = np.maximum(w, 1e-12)
        self.p = (w / w.sum()).tolist()

        self.drop_last = bool(drop_last)

        # RNG
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

        # iterators cache
        self._iters: Dict[str, Iterator] = {}

    def __len__(self) -> int:
        return self.iters_per_epoch

    def _get_iter(self, name: str) -> Iterator:
        if name not in self._iters:
            self._iters[name] = iter(self.loaders[name])
        return self._iters[name]

    def _reset_iter(self, name: str):
        self._iters[name] = iter(self.loaders[name])

    def __iter__(self):
        # reset RNG each epoch for determinism (optional)
        self._rng = random.Random(self._seed)

        # reset iterators each epoch
        self._iters = {k: iter(v) for k, v in self.loaders.items()}

        for _ in range(self.iters_per_epoch):
            # choose dataset
            name = self._rng.choices(self.names, weights=self.p, k=1)[0]

            # fetch next batch
            it = self._get_iter(name)
            try:
                batch = next(it)
            except StopIteration:
                # restart iterator when exhausted
                self._reset_iter(name)
                it = self._get_iter(name)
                batch = next(it)

            # IMPORTANT: yield dataset_name so runner can switch graph state
            yield (name, batch)