import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

# from pfhedge.stochastic import generate_heston

def simulate_BM(n_sample, dt, n_timestep):
    noise = torch.randn(size = (n_sample, n_timestep))
    paths_incr = noise * torch.sqrt(torch.tensor(dt))
    paths = torch.cumsum(paths_incr, dim=1)
    BM_paths = torch.cat([torch.zeros((n_sample,1)),paths],dim = 1) 
    BM_paths = BM_paths[...,None]
    return BM_paths

def simulate_BS(n_sample, dt, n_timestep, mu, sigma):    # Maybe define it as class method? 
    time_grid = torch.linspace(0,dt*n_timestep, n_timestep+1)
    time_paths = time_grid.expand([n_sample, n_timestep+1])[...,None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    BS_paths = torch.exp(sigma * BM_paths + (mu - 0.5*sigma**2)*time_paths)  
    return BS_paths

def simulate_time(n_sample, dt, n_timestep, reverse = False):
    time_grid = torch.linspace(0,dt*n_timestep, n_timestep+1)
    time_paths = time_grid.expand([n_sample, n_timestep+1])[...,None]
    if reverse: 
        return n_timestep*dt - time_paths
    else:
        return time_paths


class BlackScholesDataset(TensorDataset):
    def __init__(self, 
                 n_sample, 
                 n_timestep,  
                 dt,
                 mu = 0.0, 
                 sigma = 0.2,
                 ):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.prices = simulate_BS(self.n_sample, self.dt, self.n_timestep, self.mu, self.sigma).type(torch.float32)
        self.paths = self.prices
        super(BlackScholesDataset, self).__init__(self.paths)
    
# class HestonDataset(TensorDataset):
#     def __init__(self, 
#                  n_sample, 
#                  n_timestep,
#                  dt,
#                  kappa: float = 1.0,
#                  theta: float = 0.04,
#                  sigma: float = 0.2,
#                  rho: float = -0.7,
#                  ):
#         self.kappa = kappa
#         self.theta = theta
#         self.sigma = sigma
#         self.rho = rho
#         self.dt = dt
#         self.n_sample = n_sample
#         self.n_timestep = n_timestep
#         self.inital_values = (1. , theta)
#         self.paths = self.simulate_Heston().type(torch.float32)
#         super(HestonDataset, self).__init__(self.paths)

#     def simulate_Heston(self):
#         hestontuple = generate_heston(
#             n_paths = self.n_sample,
#             n_steps = self.n_timestep+1,
#             init_state = self.inital_values,
#             kappa = self.kappa,
#             theta = self.theta,
#             sigma = self.sigma,
#             rho = self.rho,
#             dt = self.dt)
#         self.prices = hestontuple.spot[...,None]
#         self.variances = hestontuple.variance[...,None]
#         self.paths = torch.cat([self.prices, self.variances], dim = -1)
#         return self.paths
    
#     def calculate_varswap(self):
#         self.times_inverse = simulate_time(self.n_sample, self.dt, self.n_timestep, reverse = True)
#         self.prices_varswap = torch.cumsum(self.variances,dim=1) * self.dt + self.L_func(self.times_inverse, self.variances)
#         return self.prices_varswap

#     def L_func(self, tau: Tensor, v: Tensor) -> Tensor:
#         L = (v-self.theta) / self.kappa * (1-(-self.kappa*(tau)).exp()) + self.theta*tau
#         return L
    

        
        
        
        


        
