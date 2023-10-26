
from neuralhedge.data.stochastic import BlackScholesDataset, simulate_time
import torch

def stockbond_market(mu, sigma, r, n_sample = 5000, n_timestep = 60, dt = 1/12):
    ds_bs = BlackScholesDataset(n_sample = n_sample,
                        n_timestep = n_timestep,
                        dt = dt,
                        mu = mu,
                        sigma = sigma)
    time = simulate_time(n_sample, dt, n_timestep, reverse = False)
    bond = torch.exp(r*time)
    paths = torch.cat([ds_bs.paths, bond], dim = -1)

    info1 = torch.log(paths)
    info2 = simulate_time(n_sample, dt, n_timestep, reverse = True)
    info = torch.cat([info1,
                info2],
                dim = -1)
    data = [paths, info]
    return data 