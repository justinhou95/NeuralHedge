
from neuralhedge.data.stochastic import BlackScholes, simulate_time
import torch

def stock_bond_market(mu = 0.1, 
                      sigma = 0.2, 
                      r = 0.01, 
                      n_sample = 5000, 
                      n_timestep = 60, 
                      dt = 1/12):
    # Stock
    bs = BlackScholes(n_sample = n_sample,
                        n_timestep = n_timestep,
                        dt = dt,
                        mu = mu,
                        sigma = sigma)
    # Bond
    time = simulate_time(n_sample, dt, n_timestep, reverse = False)
    bond = torch.exp(r*time)
    # Prices
    prices = torch.cat([bs.prices, bond], dim = -1)
    # Information 
    info1 = torch.log(prices)
    info2 = simulate_time(n_sample, dt, n_timestep, reverse = True)
    info = torch.cat([info1,
                info2],
                dim = -1)
    return prices, info