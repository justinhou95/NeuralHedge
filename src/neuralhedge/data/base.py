from typing import List
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
    def __init__(self, 
                 prices: Tensor,
                 information: Tensor,
                 payoff: Tensor):
        self.prices = prices
        self.information = information 
        self.payoff = payoff
        # TODO: check len(prices) == len(information) == len(payoff)
        # TODO: check shape
        # TODO: check input type

    def __len__(self):
        return len(self.prices) 

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.information[idx], self.payoff[idx]]
    

class ManagerDataset(HedgerDataset):
    """Market information dataset.
    Args:
        - prices, information
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
    """
    def __init__(self, 
                 prices: Tensor,
                 information: Tensor,
                 ):
        self.prices = prices
        self.information = information 

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.information[idx]]