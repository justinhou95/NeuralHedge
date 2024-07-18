from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset


class HedgerDataset(Dataset):
    """Market information dataset.
    Args:
        - prices, information, payoff
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
        - payoff: (n_samples,)
    """

    def __init__(self, prices: Tensor, info: Tensor, payoff: Tensor):
        self.data = (prices, info, payoff)
        self.prices = prices
        self.info = info
        self.payoff = payoff
        # TODO: check shape
        # TODO: check input type

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.info[idx], self.payoff[idx]]


class ManagerDataset(Dataset):
    """Market information dataset.
    Args:
        - prices, information
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
    """

    def __init__(
        self,
        prices: Tensor,
        info: Tensor,
    ):
        self.data = (prices, info)
        self.prices = prices
        self.info = info

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.info[idx]]
