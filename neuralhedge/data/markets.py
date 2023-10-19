
import numpy as np
import torch

from math import ceil
import matplotlib.pyplot as plt

from neuralhedge.nn import datahedger, contigent
from neuralhedge.data import stochastic
from neuralhedge._utils.plotting import plot_data_set

def BlackScholesTestMarket(deep = False):
    n_paths = 50000
    step_size = 1/365
    maturity = 30/365
    n_steps = ceil(maturity / step_size)
    initial_value = 100.
    strike = initial_value
    mu = 0.0
    sigma = 0.2

    blackscholes = stochastic.BlackScholes(mu = mu, 
                                sigma = sigma,
                                n_paths = n_paths,
                                n_steps = n_steps,
                                step_size = step_size)
    blackscholes.stimulate(initial_value = initial_value)
    option = contigent.EuropeanVanilla(strike = strike)

    paths = blackscholes.prices     # (n_paths, n_steps+1, n_asset)
    if deep:
        information = torch.log(blackscholes.prices)
    else:
        information = torch.cat([
            torch.log(blackscholes.prices),
            blackscholes.times_inverse
            ], axis = -1)
        
    payoff = option.payoff(blackscholes.prices)   # (n_paths, n_steps+1, 1)

    n_asset = paths.shape[-1]
    n_feature = information.shape[-1]
    data_set = [paths, information, payoff]


    dataset_market = datahedger.MarketDataset(data_set)

    return data_set, dataset_market

def HestonTestMarket():
    n_paths = 50000
    step_size = 1/365
    maturity = 30/365
    n_steps = ceil(maturity / step_size)

    kappa = 1.
    theta = 0.04
    sigma = 0.2
    rho = -0.7

    v0 = 0.04
    s0 = 100.
    initial_value = (s0, v0)

    heston = stochastic.Heston(kappa = kappa,
                                theta = theta,
                                sigma = sigma,
                                rho = rho,
                                n_paths = n_paths,
                                n_steps = n_steps,
                                step_size = step_size)
    heston.stimulate(initial_value = initial_value)
    option = contigent.EuropeanVanilla(strike = s0)

    paths = torch.cat([heston.prices, heston.prices_varswap],axis = -1)
    information = torch.cat([
        torch.log(heston.prices),
        heston.variances,
        heston.times_inverse
        ], axis = -1)
    payoff = option.payoff(heston.prices)

    data_set = [paths, information, payoff]
    dataset_market = datahedger.MarketDataset(data_set)
    

    return data_set, dataset_market




