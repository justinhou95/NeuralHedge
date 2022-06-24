from typing import Tuple

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge.stochastic import generate_heston, generate_geometric_brownian

class StochasticProcesses(ABC):
    def __init__(
        self, 
        n_paths: int,
        n_steps: int,
        step_size: float) -> None:

        self._n_paths = n_paths
        self._n_steps = n_steps
        self._step_size = step_size
        self._time = torch.arange(0, n_steps+1) * step_size
        self._times = torch.tile(self._time, [n_paths,1])[...,None]
        self._time_inverse = torch.linspace(n_steps,0,n_steps+1) * step_size
        self._times_inverse = torch.tile(self._time_inverse, [n_paths,1])[...,None]

    @property
    def n_paths(self):
        return self._n_paths

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def step_size(self):
        return self._step_size

    @property
    def times(self):
        return self._times

    @property
    def times_inverse(self):
        return self._times_inverse

    @property
    def initial_values(self):
        return self._initial_value
    
    @property
    def paths(self):
        return self._paths

    @abstractmethod
    def stimulate(self):
        pass

class BlackScholes(StochasticProcesses, Module):
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        **kwarg
        ):
        super().__init__(**kwarg)
        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self):
        return self._mu
    @property
    def sigma(self):
        return self._sigma    
    @property
    def parameter(self):
        return self._mu, self._sigma


    @property
    def prices(self):
        return self._paths   

    def stimulate(self, initial_value: float) -> Tensor:
        self._paths = generate_geometric_brownian(
            n_paths = self.n_paths, 
            n_steps = self.n_steps+1, 
            init_state = initial_value,
            mu = self.mu,
            sigma = self.sigma,
            dt = self.step_size)[..., None]
        return self.paths


class Heston(StochasticProcesses, Module):
    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.2,
        rho: float = -0.7,
        **kwarg
        ):
        super().__init__(**kwarg)
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma
        self._rho = rho

    # Parameters
    @property
    def kappa(self):
        return self._kappa
    @property
    def theta(self):
        return self._theta
    @property
    def sigma(self):
        return self._sigma
    @property
    def rho(self):
        return self._rho
    @property
    def parameter(self):
        return self.kappa, self._theta, self._sigma, self._rho
        
    # Paths 
    @property
    def prices(self):
        return self._paths[...,0][...,None]
    @property
    def variances(self):
        return self._paths[...,1][...,None] 
    @property
    def prices_varswap(self):
        return self._prices_varswap


    def stimulate(self, initial_value: Tuple[float], prices_varswap = True) -> Tensor:
        # maturity = n_steps * step_size
        # time_to_maturity = torch.linspace(maturity, 0, n_steps + 1)
        # time_to_maturity = torch.tile(time_to_maturity, [n_paths,1])[...,None]

        self._initial_value = initial_value
        hestontuple = generate_heston(
            n_paths = self.n_paths,
            n_steps = self.n_steps+1,
            init_state = self.initial_values,
            kappa = self.kappa,
            theta = self.theta,
            sigma = self.sigma,
            rho = self.rho,
            dt = self.step_size,)

        self._prices = hestontuple.spot[...,None]
        self._variances = hestontuple.variance[...,None]
        self._paths = torch.cat([self._prices, self._variances],axis = -1)
        if prices_varswap:
            self._prices_varswap = torch.cumsum(self.variances,dim=1) * self.step_size + self.L_func(self.times_inverse, self.variances)

        return self.paths

    def L_func(self, tau: Tensor, v: Tensor) -> Tensor:
        L = (v-self.theta) / self.kappa * (1-(-self.kappa*(tau)).exp()) + self.theta*tau
        return L
    

        
        
        
        


        
