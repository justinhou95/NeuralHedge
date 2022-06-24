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
    def payoff(self, paths: Tensor) -> Tensor:
        if self.call:
            return F.relu(paths[:,-1,:] - self.strike)
        else:
            return F.relu(self.strike - paths[:,-1,:])
