import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset


def simulate_BM(n_sample, dt, n_timestep):
    noise = torch.randn(size=(n_sample, n_timestep))
    paths_incr = noise * torch.sqrt(torch.tensor(dt))
    paths = torch.cumsum(paths_incr, dim=1)
    BM_paths = torch.cat([torch.zeros((n_sample, 1)), paths], dim=1)
    BM_paths = BM_paths[..., None]
    return BM_paths


def simulate_BS(
    n_sample, dt, n_timestep, mu, sigma
):  # Maybe define it as class method?
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    BS_paths = torch.exp(sigma * BM_paths + (mu - 0.5 * sigma**2) * time_paths)
    return BS_paths


def simulate_time(n_sample, dt, n_timestep, reverse=False):
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    if reverse:
        return n_timestep * dt - time_paths
    else:
        return time_paths


class BlackScholes:
    def __init__(self, n_sample, n_timestep, dt, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.n_sample = n_sample
        self.n_timestep = n_timestep

    def get_prices(self):
        self.prices = simulate_BS(
            self.n_sample, self.dt, self.n_timestep, self.mu, self.sigma
        ).type(torch.float32)
        return self.prices
