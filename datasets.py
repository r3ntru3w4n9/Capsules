r"""
Wrapper datasets for your target dataset

:class: `DelegateDataset`

* wraps a dataset, lazily move dataset output to `device` recursively

:class: `CacheDataset`

* wraps a dataset, move dataset output to `device` recursively once and for all
"""
import torch
from torch.utils.data import Dataset

import functions


class DelegateDataset(Dataset):
    def __init__(self, dataset: Dataset, device: str):
        super().__init__()
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return functions.to_device(iterable=self.dataset[idx], device=self.device)


class CacheDataset(Dataset):
    def __init__(self, dataset: Dataset, device: str):
        super().__init__()
        self.data = functions.to_device(iterable=dataset, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
