import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

class EuropeanVanilla(Module):
    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0
        ):
        super().__init__()
        self.call = call
        self.strike = strike
    def payoff(self, prices: Tensor) -> Tensor:
        """Returns the terminal payoff of financial derivative
            Shape: payoff: (n_sample, 1) 
        """
        if self.call:
            payoff = F.relu(prices - self.strike)
        else:
            payoff = F.relu(self.strike - prices)
        return payoff[...,None] 
        
    # def payoff_all(self, paths: Tensor) -> Tensor:
    #     """Returns the payoff of financial derivative
    #         Shape: payoff_all: (n_sample, n_timestep+1, 1) 
    #     """
    #     payoff_all = torch.zeros(paths.shape[:2])
    #     if self.call:
    #         payoff_all[:,-1] = F.relu(paths[:,-1,0] - self.strike)
    #     else:
    #         payoff_all[:,-1] = F.relu(self.strike - paths[:,-1,0])
    #     return payoff_all[...,None] 
        
