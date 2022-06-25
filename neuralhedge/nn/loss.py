
from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from abc import ABC, abstractmethod

def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    """
    f(X) = -\exp(-aX) 
    """
    return -(-a * input).exp()

def value_at_risk(input: Tensor, q: float = 0.01) -> Tensor:
    return torch.quantile(input, q, interpolation='linear')

def expected_shortfall(input: Tensor, q: float = 0.01) -> Tensor:
    VaR = value_at_risk(input, q)
    ES = F.relu(VaR - input).mean()/q - VaR
    return ES


class LossMeasure(ABC, Module):

    @abstractmethod
    def cash(self,):
        pass


class EntropicRiskMeasure(LossMeasure):

    @property
    def a(self):
        return self._a
    @a.setter
    def a(self,a: float = 1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")
        self._a = a
    
    def __init__(self, a: float = 1.0) -> None:
        super().__init__()
        self.a = a

    def forward(self, input: Tensor) -> Tensor:
        """
        f(X) = (1/a) * log(\E[exp(-aX)])
        """
        return (-exp_utility(input - input.min(), a=self.a).mean(0)).log() / self.a - input.min()

    def cash(self, input: Tensor) -> Tensor:
        """
        f(X) = (1/a) * log(a\E[exp(-aX)])
        """
        return (-exp_utility(input - input.min(), a=self.a).mean(0) * self.a).log() / self.a - input.min()


class SquareMeasure(LossMeasure):
    @property
    def a(self):
        return self._a
    
    def __init__(self, a: float = 1.0) -> None:
        super().__init__()
        self._a = a

    def forward(self, input: Tensor) -> Tensor:
        """
        f(X) = Var(X)/2 - E[X]
        """
        return input.var()/2 - input.mean()

    def cash(self, input: Tensor) -> Tensor:
        """
        f(X) = -E[X]
        """
        return -input.mean()

class ExpectedShortfall(LossMeasure):
    @property
    def q(self):
        return self._q
    @q.setter
    def q(self,q: float = 0.5):
        if not q > 0 or not q < 1:
            raise ValueError("Risk aversion coefficient should be between 0 and 1")
        self._q = q
    
    def __init__(self, q: float = 0.5) -> None:
        super().__init__()
        self._q = q

    def forward(self, input: Tensor) -> Tensor:
        """
        f(X) = ES(X)
        """
        return expected_shortfall(input, self.q)

    def cash(self, input: Tensor) -> Tensor:
        """
        f(X) = -E[X]
        """
        return -value_at_risk(input, self.q)


def proportional_cost(holding_diff, price_now) -> Tensor:
    cost = 0.001 * torch.abs(holding_diff) * price_now
    return cost

def no_cost(holding_diff, price_now) -> Tensor:
    cost = 0.
    return cost