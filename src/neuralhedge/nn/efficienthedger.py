from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from neuralhedge.nn.datahedger import Hedger
from os import path as pt

class EfficientHedger(Hedger):
    def __init__(self,model: Module , init_wealth = 0., ad_bound = 0.):
        super().__init__(model)
        self.init_wealth = init_wealth
        self.ad_bound = ad_bound

    def compute_info(self, holding: Tensor, info: Tensor, t = None) -> Tensor:
        all_info = info[:, t, :]
        return all_info
    
    def compute_loss(self, input: List[Tensor]):
        prices, information, payoff = input
        wealth = self.forward(input)
        wealth_tensor = self.init_wealth + torch.stack(wealth,dim = 1)  # (batch,time)
        pnl = wealth_tensor[:,-1] - payoff 
        ad_cost = admissible_cost(wealth_tensor, self.ad_bound)
        loss = self.risk(pnl) + ad_cost
        return loss
    
