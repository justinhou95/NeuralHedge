from os import path as pt
from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from neuralhedge.nn.datahedger import Hedger
from neuralhedge.nn.loss import log_utility


class Manager(Hedger):
    def __init__(self, strategy: Module, utility_func=log_utility):
        super().__init__(strategy)
        self.utility_func = utility_func

    def forward(self, prices: Tensor, info: Tensor):
        batch_size = prices.shape[0]
        wealth = [torch.ones([batch_size]) for t in range(prices.shape[1])]
        prop_hold = torch.zeros_like(prices)
        for t in range(prices.shape[1] - 1):
            info_dyn = wealth[t]
            all_info_t = self.compute_info_t(info_dyn, info, t)
            prop_hold[:, t + 1, :] = self.compute_prop_hold_tplus1(all_info_t, t)
            return_t = torch.sum(
                prop_hold[:, t + 1, :] * (prices[:, t + 1, :] / prices[:, t, :]),
                dim=-1,
                keepdim=False,
            )
            wealth[t + 1] = wealth[t] * return_t
        wealth = torch.cat([wealth_t.unsqueeze(1) for wealth_t in wealth], dim=1)
        return wealth

    def compute_prop_hold_tplus1(self, all_info_t: Tensor, t=None) -> Tensor:
        prop_hold_tplus1 = self.strategy(all_info_t)
        return prop_hold_tplus1

    def compute_info_t(self, info_dyn: Tensor, info: Tensor, t=None) -> Tensor:
        all_info_t = info[:, t, :]
        return all_info_t

    def compute_loss(self, input: List[Tensor]):
        prices, info = input
        terminal_wealth = self.forward(prices, info)[-1]
        return -self.utility_func(terminal_wealth)

    def record_history(self):
        return self.history["alpha"].append(list(self.parameters())[0].item())


class WealthManager(Manager):
    """
    Last coordinate of all_info_t is wealth
    """

    def __init__(self, model: Module, utility_func=...):
        super().__init__(model, utility_func)

    def compute_info_t(self, info_dyn: Tensor, info: Tensor, t=None) -> Tensor:
        info_dyn = info_dyn.unsqueeze(1)
        all_info_t = torch.cat([info[:, t, :], info_dyn], dim=-1)
        return all_info_t
