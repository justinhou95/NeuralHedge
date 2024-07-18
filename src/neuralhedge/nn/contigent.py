from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F


class EuropeanVanilla(Module):
    def __init__(
        self,
        strike: float,
        call: bool = True,
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
        return payoff
