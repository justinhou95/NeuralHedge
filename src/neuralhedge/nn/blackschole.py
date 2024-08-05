"""
Benchmark strategy for BlackScholes

"""

import torch
from torch.distributions.normal import Normal
from torch.nn import Module


class BlackScholesPrice(Module):
    r"""
    Pricing kernel of BlackScholes model

    """

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
        self.normal = Normal(0, 1)

    def forward(self, x):
        r"""
        Arguments:
            x (:class:`torch.Tensor`): (log_price = x[..., 0], time_to_maturity = x[..., 1])

        Returns:
            bs_price (:class:`torch.Tensor`):
        """
        log_price = x[..., 0]
        time_to_maturity = x[..., 1]
        price = log_price.exp()
        d1 = (
            log_price
            - self.strike.log()
            + (self.r + self.sigma**2 / 2) * time_to_maturity
        ) / (self.sigma * time_to_maturity.sqrt())
        d2 = d1 - (self.sigma * time_to_maturity.sqrt())
        output = price * self.normal.cdf(d1) - self.strike * self.normal.cdf(
            d2
        ) * torch.exp(-self.r * time_to_maturity)
        bs_price = output[..., None]
        return bs_price


class BlackScholesDelta(Module):
    r"""
    Delta hedging Model
    """

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
        self.normal = Normal(0, 1)

    def forward(self, x):
        r"""
        Arguments:
            x (:class:`torch.Tensor`): (log_price = x[..., 0], time_to_maturity = x[..., 1])

        Returns:
            bs_delta (:class:`torch.Tensor`):
        """
        log_price = x[..., 0]
        time_to_maturity = x[..., 1]
        d1 = (
            log_price
            - self.strike.log()
            + (self.r + self.sigma**2 / 2) * time_to_maturity
        ) / (self.sigma * time_to_maturity.sqrt())
        bs_delta = self.normal.cdf(d1)[..., None]
        return bs_delta


class BlackScholesAlpha(Module):
    r"""
    Merton problem Model
    """

    def __init__(self, mu, sigma, r, alpha=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.r = r
        if alpha is None:
            self.alpha = (self.mu - self.r) / (self.sigma**2)
        else:
            self.alpha = alpha

    def forward(self, x):
        r"""
        Returns:
            prop (:class:`torch.Tensor`):
        Shape:
            prop: (n_sample, 2)
        """
        prop1 = torch.ones_like(x[..., :1]) * self.alpha
        prop2 = torch.ones_like(x[..., :1]) * (1 - self.alpha)
        prop = torch.cat([prop1, prop2], dim=-1)
        return prop


class BlackScholesMeanVarianceAlpha(BlackScholesAlpha):
    r"""
    Mean Variance Model
    """

    def __init__(self, mu, sigma, r, Wstar):
        super().__init__(mu, sigma, r)
        self.xi = (self.mu - self.r) / self.sigma
        self.Wstar = Wstar

    def compute_alpha(self, x):
        wealth = x[..., -1]
        time_to_maturity = x[..., 1]
        alpha = (
            self.xi
            / self.sigma
            / wealth
            * (self.Wstar * torch.exp(-self.r * time_to_maturity) - wealth)
        )
        return alpha.view(-1, 1)

    def forward(self, x):
        r"""
        Returns:
            prop (:class:`torch.Tensor`):
        Shape:
            prop: (n_sample, 2)
        """
        alpha = self.compute_alpha(x)
        prop = torch.cat([alpha, 1 - alpha], dim=-1)
        return prop


class BlackScholesMeanVarianceAlphaClip(BlackScholesMeanVarianceAlpha):
    r"""
    Mean Variance Clipped Model
    """

    def __init__(self, mu, sigma, r, Wstar, clip):
        super().__init__(mu, sigma, r, Wstar)
        self.clip = clip

    def forward(self, x):
        r"""
        Returns:
            prop (:class:`torch.Tensor`):
        Shape:
            prop: (n_sample, 2)
        """
        alpha = self.compute_alpha(x)

        alpha = alpha * (alpha > 0)
        alpha_clip = alpha * (alpha < self.clip) + self.clip * (alpha >= self.clip)

        prop = torch.cat([alpha_clip, 1 - alpha_clip], dim=-1)
        return prop
