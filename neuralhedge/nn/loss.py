
from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    return -(-a * input).exp()

def entropic_risk_measure(input: Tensor, a: float = 1.0) -> Tensor:
    return (-exp_utility(input - input.min(), a=a).mean(0)).log() / a - input.min()

class EntropicRiskMeasure(Module):
    
    def __init__(self, a: float = 1.0) -> None:
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def forward(self, input: Tensor, target: Union[Tensor, float, int] = 0) -> Tensor:
        return entropic_risk_measure(input - target, a=self.a)

    def cash(self, input: Tensor, target: Union[Tensor, float, int] = 0) -> Tensor:
        return -self(input - target)

