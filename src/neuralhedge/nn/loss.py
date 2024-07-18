from typing import Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def proportional_cost(holding_diff, price_now) -> Tensor:
    cost = 0.001 * torch.abs(holding_diff) * price_now
    return cost


def admissible_cost(wealth, bound=0.0) -> Tensor:
    cost = F.relu(bound - torch.min(wealth, dim=1).values).mean()
    return cost


def no_cost(holding_diff, price_now) -> Tensor:
    cost = torch.tensor(0.0)
    return cost


def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    """
    f(X) = -exp(-aX)
    """
    return -(-a * input).exp()


def log_utility(x: Tensor) -> Tensor:
    return x.log().mean()


def value_at_risk(input: Tensor, q: float = 0.01) -> Tensor:
    return torch.quantile(input, q, interpolation="linear")


def expected_shortfall(input: Tensor, q: float = 0.01) -> Tensor:
    VaR = value_at_risk(input, q)
    ES = F.relu(VaR - input).mean() / q - VaR
    return ES


class LossMeasure(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class PowerMeasure(LossMeasure):
    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p: float = 1.0):
        if not p > 0:
            raise ValueError("Risk aversion coefficient should be positive.")
        self._p = p

    def __init__(self, p: float = 1.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """
        f(X) = (1/p) (E[max{X,0}^{p}])
        """
        input_T = input
        return (F.relu(-input_T) ** self.p).mean() ** (1 / self.p) / self.p


class EntropicRiskMeasure(LossMeasure):
    @property  # need property here?
    def a(self):
        return self._a

    @a.setter
    def a(self, a: float = 1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")
        self._a = a

    def __init__(self, a: float = 1.0) -> None:
        super().__init__()
        self.a = a

    def forward(self, input_T: Tensor) -> Tensor:
        """
        rho(X) = (1/a) * log(E[exp(-aX)])
        """
        input_T_min = input_T.min()
        return (
            -exp_utility(input_T - input_T_min, a=self.a).mean()
        ).log() / self.a - input_T_min

    def optimal_omega(self, input_T: Tensor) -> Tensor:
        """
        input: (n_paths, n_steps+1, 1)
        f(X) = (1/a) * log(aE[exp(-aX)])
        """
        input_T_min = input_T.min()
        return (
            -exp_utility(input_T - input_T_min, a=self.a).mean() * self.a
        ).log() / self.a - input_T_min


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
        input_T = input
        return input_T.var() / 2 - input_T.mean()

    def optimal_omega(self, input: Tensor) -> Tensor:
        """
        f(X) = -E[X]
        """
        input_T = input
        return -input_T.mean()


class ExpectedShortfall(LossMeasure):
    "q = 1-alpha"

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q: float = 0.5):
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
        input_T = input
        return expected_shortfall(input_T, self.q)

    def optimal_omega(self, input: Tensor) -> Tensor:
        """
        f(X) = -VaR_q(X)
        """
        input_T = input
        return -value_at_risk(input_T, self.q)
