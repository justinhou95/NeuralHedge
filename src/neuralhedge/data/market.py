import torch

from neuralhedge.data.base import HedgerDataset, ManagerDataset
from neuralhedge.data.stochastic import BlackScholes, simulate_time
from neuralhedge.nn import blackschole, datahedger
from neuralhedge.nn.contigent import EuropeanVanilla


class BS_Market:
    """
    Data of BS stock + Bond + European call option
    For Portfolio Management, the option part is redundant
    """

    def __init__(
        self,
        n_sample=10000,
        n_timestep=30,
        dt=1 / 30,
        mu=0.1,
        sigma=0.2,
        r=0.0,
        init_price=100.0,
    ) -> None:
        self.n_sample = n_sample
        self.n_timestep = n_timestep
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = n_timestep * self.dt
        self.bs = BlackScholes(
            n_sample=n_sample, n_timestep=n_timestep, dt=dt, mu=mu, sigma=sigma
        )

        self.init_price = init_price

        self.strike = 100.0
        self.contigent = EuropeanVanilla(strike=self.strike, call=True)

        bs_pricer = blackschole.BlackScholesPrice(
            sigma=sigma, risk_free_rate=r, strike=self.contigent.strike
        )

        tmp = torch.tensor([torch.tensor(self.init_price).log(), self.T])
        self.bs_price = bs_pricer(tmp)[0]
        self.bs_delta = blackschole.BlackScholesDelta(
            sigma=sigma, risk_free_rate=r, strike=self.contigent.strike
        )

    def get_hedge_ds(self):
        stock_prices = self.bs.get_prices() * self.init_price
        time = simulate_time(self.n_sample, self.dt, self.n_timestep, reverse=False)
        bond_prices = torch.exp(self.r * time) * self.init_price
        prices = torch.cat([stock_prices, bond_prices], dim=-1)
        payoff = self.contigent.payoff(prices[:, -1, 0])
        info = torch.cat(
            [
                torch.log(prices[..., :1]),
                torch.sqrt(
                    simulate_time(self.n_sample, self.dt, self.n_timestep, reverse=True)
                ),
            ],
            dim=-1,
        )
        data = (prices, info, payoff)
        hedge_ds = HedgerDataset(*data)
        return hedge_ds

    def get_manage_ds(self):
        stock_prices = self.bs.get_prices() * self.init_price
        time = simulate_time(self.n_sample, self.dt, self.n_timestep, reverse=False)
        bond_prices = torch.exp(self.r * time) * self.init_price
        prices = torch.cat([stock_prices, bond_prices], dim=-1)
        # TODO: info in dict form, now 1st log price, second time to maturity
        info = torch.cat(
            [
                torch.log(prices[..., :1]),
                simulate_time(self.n_sample, self.dt, self.n_timestep, reverse=True),
            ],
            dim=-1,
        )
        data = (prices, info)
        manage_ds = ManagerDataset(*data)
        return manage_ds

    def get_price_delta(self):
        return self.bs_price, self.bs_delta


# def stock_bond_market(mu = 0.1,
#                       sigma = 0.2,
#                       r = 0.01,
#                       n_sample = 5000,
#                       n_timestep = 60,
#                       dt = 1/12):
#     # Stock
#     bs = BlackScholes(n_sample = n_sample,
#                         n_timestep = n_timestep,
#                         dt = dt,
#                         mu = mu,
#                         sigma = sigma)
#     # Bond
#     time = simulate_time(n_sample, dt, n_timestep, reverse = False)
#     bond = torch.exp(r*time)
#     # Prices
#     prices = torch.cat([bs.prices, bond], dim = -1)
#     # Information
#     info1 = torch.log(prices)
#     info2 = simulate_time(n_sample, dt, n_timestep, reverse = True)
#     info = torch.cat([info1,
#                 info2],
#                 dim = -1)
#     return prices, info
