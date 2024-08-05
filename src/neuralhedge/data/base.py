from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset


class HedgerDataset(Dataset):
    r"""
    Dataset contains data for hedging

    Args:
        prices (:class:`torch.Tensor`)
        information (:class:`torch.Tensor`)
        payoff (:class:`torch.Tensor`)
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
        - payoff: (n_sample,)

    Attributes:
        data (:class:`tuple`): (prices, information, payoff)

    """

    def __init__(self, prices: Tensor, info: Tensor, payoff: Tensor):
        self.data = (prices, info, payoff)
        self.prices = prices
        self.info = info
        self.payoff = payoff
        # TODO: check shape
        # TODO: check input type
        # TODO: data from tuple to dict

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.info[idx], self.payoff[idx]]


class ManagerDataset(Dataset):
    r"""
    Dataset contains data for management

    Args:
        prices (:class:`torch.Tensor`)
        information (:class:`torch.Tensor`)
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)

    Attributes:
        data (:class:`tuple`): (prices, information)

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
