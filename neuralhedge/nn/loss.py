
from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from abc import ABC, abstractmethod

def proportional_cost(holding_diff, price_now) -> Tensor:
    cost = 0.001 * torch.abs(holding_diff) * price_now
    return cost

def no_cost(holding_diff, price_now) -> Tensor:
    cost = 0.
    return cost

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
        input_T = input[:,-1,:]
        input_T_min = input_T.min()
        return (-exp_utility(input_T - input_T_min, a=self.a).mean(0)).log() / self.a - input_T_min

    def cash(self, input: Tensor) -> Tensor:
        """
        input: (n_paths, n_steps+1, 1)
        f(X) = (1/a) * log(a\E[exp(-aX)])
        """
        input_T = input[:,-1,:]
        input_T_min = input_T.min()
        return (-exp_utility(input_T - input_T_min, a=self.a).mean(0) * self.a).log() / self.a - input_T_min


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
        input_T = input[:,-1,:]
        return input_T.var()/2 - input_T.mean()

    def cash(self, input: Tensor) -> Tensor:
        """
        f(X) = -E[X]
        """
        input_T = input[:,-1,:]
        return -input_T.mean()

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

    def l_func(self, input: Tensor) -> Tensor:
        return F.relu(-input) / self.q

    def forward(self, input: Tensor) -> Tensor:
        """
        f(X) = ES_q(X)
        """
        input_T = input[:,-1,:]
        return expected_shortfall(input_T, self.q)

    def cash(self, input: Tensor) -> Tensor:
        """
        f(X) = -VaR_q(X)
        """
        input_T = input[:,-1,:]
        return -value_at_risk(input_T, self.q)


class OCELoss(LossMeasure):

    def __init__(self, loss = ExpectedShortfall()) -> None:
        super().__init__()
        self.oce = True
        self.loss = loss.l_func
        
    def forward(self, input: Tensor, wealth_0: Tensor) -> Tensor:
        input_T = input[:,-1,:]
        return torch.mean(wealth_0 + 100* input_T**2)
        
    def cash(self, input: Tensor) -> Tensor:
        input_T = input[:,-1,:]
        return 0