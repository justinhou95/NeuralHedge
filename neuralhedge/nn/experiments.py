import torch
from neuralhedge.data.stochastic import BlackScholesDataset, simulate_time
import numpy as np 
from neuralhedge.nn.blackschole import BlackScholesAlpha, BlackScholesMeanVarianceAlpha, BlackScholesMeanVarianceAlphaClip
from neuralhedge.nn.datamanager import WealthManager
from neuralhedge.data.market import stockbond_market

def BS_efficient_frontier(mu, sigma, r, data = None):   
    """ Calculate the efficient frontier
    Args:
        - data: if data is None, we calculate the efficient frontier of Black-Scholes data under constant, optimal, and optimal_clipped strategy for the mean variance portfolio optimization problem i.e. max E[V_T] - kappa*Var(V_T). If data is not None, we calculate the efficient frontier of data under the constant, optimal, and optimal_clipped strategy (optimal in the sense of under BlackScholes with parameter mu, sigma, r)
    """ 
    if data is None: 
        data = stockbond_market(mu,sigma,r)
    
    strategy_dict = {}
    mean_list = []
    std_list = []
    model = BlackScholesAlpha(mu, sigma, r)
    manager = WealthManager(model)
    for alpha in np.linspace(0,1,101):
        model = BlackScholesAlpha(mu, sigma, r, alpha = alpha)
        manager.model = model
        terminal_wealth = manager(data)[-1]
        mean_list.append(terminal_wealth.mean().item())
        std_list.append(terminal_wealth.std().item())
    strategy_dict['constant'] = (std_list, mean_list)


    mean_list = []
    std_list = []
    model = BlackScholesAlpha(mu, sigma, r)
    manager = WealthManager(model)
    for Wstar in np.linspace(1,4,101):
        model = BlackScholesMeanVarianceAlpha(mu, sigma, r, Wstar = Wstar)
        manager.model = model
        terminal_wealth = manager(data)[-1]
        mean_list.append(terminal_wealth.mean().item())
        std_list.append(terminal_wealth.std().item())
    strategy_dict['optimal'] = (std_list, mean_list)


    mean_list = []
    std_list = []
    model = BlackScholesAlpha(mu, sigma, r)
    manager = WealthManager(model)
    for Wstar in np.linspace(1,4,101):
        model = BlackScholesMeanVarianceAlphaClip(mu, sigma, r, Wstar = Wstar)
        manager.model = model
        terminal_wealth = manager(data)[-1]
        mean_list.append(terminal_wealth.mean().item())
        std_list.append(terminal_wealth.std().item())
    strategy_dict['optimal_clip'] = (std_list, mean_list)
    return strategy_dict
