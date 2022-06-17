from ast import Mod
from typing import List
from typing import Optional

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module

def generate_brownian_motion(
    n_paths: int,
    n_steps: int,
    step_size: float,
    n_dim: int = 1) -> Tensor:

    initial_value = torch.zeros(size = [n_paths,1,n_dim])
    noise = torch.randn(size = [n_paths,n_steps,n_dim]) * np.sqrt(step_size)
    noise_cumsum = torch.cumsum(noise, axis=1)
    brownian_motion = torch.cat([initial_value,noise_cumsum], axis=1)
    return brownian_motion  # (n_paths, n_steps+1, n_dim)

class BlackScholes(Module):
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2
        ):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    def stimulate(self,
        n_paths = int,
        n_steps = int,
        step_size = float,
        initial_price = float) -> Tensor:
        time = torch.arange(n_steps+1) * step_size
        brownian_motion = generate_brownian_motion(n_paths, n_steps, step_size)
        prices = initial_price * torch.exp( 
            (self.mu - self.sigma**2/2)*time[None,:,None]
            + self.sigma * brownian_motion
            )
        return prices
