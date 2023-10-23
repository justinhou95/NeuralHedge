from typing import List
import torch
from torch import Tensor
from torch.utils.data import Dataset

class HedgerDataset(Dataset):
    """Market information dataset.
    Args:
        - data (List[Tensor]): [prices, information, payoff]
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
        - payoff: (n_samples)
    """
    def __init__(self, data: List[Tensor]):
        self.data = data
        self.paths, self.information, self.payoff = data

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        return [self.paths[idx], self.information[idx], self.payoff[idx]]
    

class ManagerDataset(Dataset):
    """Market information dataset.
    Args:
        - data (List[Tensor]): [prices, information, payoff]
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
    """
    def __init__(self, data: List[Tensor]):
        self.data = data
        self.paths, self.information = data

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        return [self.paths[idx], self.information[idx]]