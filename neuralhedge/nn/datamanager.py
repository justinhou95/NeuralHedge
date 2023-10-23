from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from abc import ABC, abstractmethod
from neuralhedge.nn.datahedger import Hedger

from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, proportional_cost, no_cost, admissible_cost, log_utility
from neuralhedge._utils.plotting import plot_pnl, plot_history
from neuralhedge.data.base import HedgerDataset

from os import path as pt

class Manager(Hedger):
    def __init__(self,model: Module,
                 utility_func = log_utility):
        super().__init__(model)
        self.utility_func = utility_func

    def forward(self, input: List[Tensor]):
        prices, information  = input 
        batch_size = prices.shape[0]
        wealth = [torch.ones([batch_size]) for t in range(prices.shape[1])]
        prop = torch.zeros_like(prices)
        for t in range(prices.shape[1]-1):
            all_information = self.compute_info(prop, information, t)   # compute information at time t 
            prop[:,t+1,:] = self.compute_prop(all_information, t)  # compute the holding at time t+1
            return_t = torch.sum(prop[:,t+1,:] * (prices[:,t+1,:]/prices[:,t,:]),dim=-1, keepdim=False)
            wealth[t+1] = wealth[t] * return_t
        return wealth
    
    def compute_prop(self, all_information: Tensor, t = None) -> Tensor: 
        prop = self.model(all_information)
        return prop

    def compute_info(self, holding: Tensor, information: Tensor, t = None) -> Tensor:
        all_information = information[:, t, :]
        return all_information 

    def compute_loss(self, input: List[Tensor]):
        terminal_wealth = self.forward(input)[-1]
        return -self.utility_func(terminal_wealth)
    