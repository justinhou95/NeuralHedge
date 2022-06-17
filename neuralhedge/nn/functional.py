import torch
import torch.nn.functional as F
from torch import Tensor


def european_payoff(
    prices: Tensor, 
    call: bool = True, 
    strike: float = 1.0) -> Tensor:
    if call:
        return F.relu(prices - strike)
    else:
        return F.relu(strike - prices)
