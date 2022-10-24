from signal import Sigmasks
from torch.nn import Module, Sequential
from torch.nn import ReLU, Linear, LazyLinear
import torch.nn.functional as F
from typing import Union, Sequence
import torch

from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.distributions.normal import Normal




class BlackScholesPrice(Module):
    def __init__(
        self, 
        sigma = 0.2, 
        risk_free_rate = 0., 
        strike = 1,
        ):
        super(BlackScholesDelta, self).__init__()
        self.sigma = torch.tensor(sigma)
        self.r = torch.tensor(risk_free_rate)
        self.strike = torch.tensor(strike)
        self.normal = Normal(0,1)

    def forward(self, x):
        log_price = x[...,0]
        time_to_maturity = x[...,1]
        d1 = (log_price - self.strike.log() + (self.r + self.sigma**2/2)*time_to_maturity) / (self.sigma * time_to_maturity.sqrt())
        d2 = d1 - (self.sigma * time_to_maturity.sqrt())
        output = self.normal.cdf(d1)[...,None] - self.strike*self.normal.cdf(d2)[...,None]
        return 

class BlackScholesDelta(Module):
    def __init__(
        self, 
        sigma = 0.2, 
        risk_free_rate = 0., 
        strike = 1,
        ):
        super(BlackScholesDelta, self).__init__()
        self.sigma = torch.tensor(sigma)
        self.r = torch.tensor(risk_free_rate)
        self.strike = torch.tensor(strike)
        self.normal = Normal(0,1)

    def forward(self, x):
        log_price = x[...,0]
        time_to_maturity = x[...,1]
        d1 = (log_price - self.strike.log() + (self.r + self.sigma**2/2)*time_to_maturity) / (self.sigma * time_to_maturity.sqrt())
        return self.normal.cdf(d1)[...,None]
    
    
        
        

