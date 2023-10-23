from signal import Sigmasks
from torch.nn import Module
import torch
from torch.nn import Module
from torch.distributions.normal import Normal

class BlackScholesPrice(Module):
    def __init__(
        self, 
        sigma, 
        risk_free_rate, 
        strike: float,
        ):
        super(BlackScholesPrice, self).__init__()
        self.sigma = torch.tensor(sigma)
        self.r = torch.tensor(risk_free_rate)
        self.strike = torch.tensor(strike)
        self.normal = Normal(0,1)

    def forward(self, x):
        log_price = x[...,0]
        time_to_maturity = x[...,1]
        price = log_price.exp()
        d1 = (log_price - self.strike.log() + (self.r + self.sigma**2/2)*time_to_maturity) / (self.sigma * time_to_maturity.sqrt())
        d2 = d1 - (self.sigma * time_to_maturity.sqrt())
        output = price * self.normal.cdf(d1) - self.strike * self.normal.cdf(d2) * torch.exp(-self.r*time_to_maturity)
        return output[...,None]

class BlackScholesDelta(Module):
    def __init__(
        self, 
        sigma, 
        risk_free_rate, 
        strike: float,
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
    

class BlackScholesAlpha(Module):
    def __init__(
        self, 
        mu,
        sigma, 
        r, 
        ):
        super(BlackScholesAlpha, self).__init__()
        self.mu = mu 
        self.sigma = sigma 
        self.r = r
        self.alpha = (self.mu-self.r)/(self.sigma**2)

    def forward(self, x):
        prop1 = torch.ones_like(x[...,:1]) * self.alpha
        prop2 = torch.ones_like(x[...,:1]) * (1-self.alpha)
        prop = torch.cat([prop1, prop2],dim = -1)
        return prop

        
        

