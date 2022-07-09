from torch.nn import Module, Sequential
from torch.nn import ReLU, Linear, LazyLinear
import torch.nn.functional as F
from typing import Union, Sequence


from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential


class NeuralNet(Module):
    def __init__(
        self, n_output: int,
        n_hidden_layers: int = 4,
        n_units: int = 32,
        activation = ReLU
    ):
        super(NeuralNet, self).__init__()
        self.activation = activation
        self.hidden_layers = [LazyLinear(n_units) for i in range(n_hidden_layers)]
        self.hidden_activations = [activation() for i in range(n_hidden_layers)]
        self.output_layer = Linear(n_units, n_output)

    def forward(self, x):
        for (layer, activation) in zip(self.hidden_layers,self.hidden_activations):
            x = activation(layer(x))
        x = self.output_layer(x)
        return x

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
