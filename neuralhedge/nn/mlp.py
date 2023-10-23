from torch.nn import Module, Sequential
from torch.nn import ReLU, Linear, LazyLinear
from copy import deepcopy
from typing import List
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
import torch

class NeuralNetSequential(Sequential):
    def __init__(
        self,
        n_output: int = 1,
        n_layers: int = 4,
        n_units: int = 32,
        activation: Module = ReLU(),
    ):
        layers: List[Module] = []
        for i in range(n_layers):
            layers.append(LazyLinear(n_units))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units, n_output))
        super().__init__(*layers)


class SingleWeight(Module):
    def __init__(self,) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
    def forward(self, x):
        prop1 = torch.ones_like(x[...,:1]) * self.weight
        prop2 = torch.ones_like(x[...,:1]) * (1-self.weight)
        prop = torch.cat([prop1, prop2],dim = -1)
        return prop

