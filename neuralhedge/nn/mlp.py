from torch.nn import Module, Sequential
from torch.nn import ReLU, Linear, LazyLinear
import torch.nn.functional as F

class NeuralNet(Module):

    def __init__(
        self,
        n_output: int = 1,
        n_hidden_layers: int = 4,
        n_units: int = 32,
        activation = F.relu
    ):
        super(NeuralNet, self).__init__()
        self.activation = activation
        self.hidden_layers = [LazyLinear(n_units) for i in range(n_hidden_layers)]
        self.output_layer = LazyLinear(n_output)
        

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x