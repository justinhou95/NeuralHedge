from typing import List

import torch
from torch import Tensor

from neuralhedge.nn.base import BaseModel
from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, admissible_cost


class Hedger(BaseModel):
    """Hedger to hedge with only data generated but not the generating class
    Args:
        - strategy (torch.nn.Module) or models (List[Module]): depending on independent neural network at each time step or the same neural network at each time
        - risk (HedgeLoss)
    """

    def __init__(
        self, strategy: torch.nn.Module, risk: LossMeasure = EntropicRiskMeasure()
    ):

        super().__init__()
        self.strategy = strategy
        self.risk = risk

    def forward(self, prices: Tensor, info: Tensor, init_wealth: Tensor):

        wealth0_dis_list = self.compute_wealth0_dis(prices, info)
        wealth0_dis = torch.cat(
            [wealth_t.unsqueeze(1) for wealth_t in wealth0_dis_list], dim=1
        )

        init_wealth_dis = init_wealth / prices[:, 0, -1]
        wealth_dis = wealth0_dis + init_wealth_dis.unsqueeze(1)
        wealth = wealth_dis * prices[:, :, -1]
        return wealth

    def compute_wealth0_dis(self, prices, info):
        """
        Compute the discounted wealth process
        """
        batch_size, n_timestep, n_asset = prices.shape
        wealth_dis = [torch.zeros([batch_size]) for t in range(n_timestep)]
        prices_dis = prices[..., :-1] / prices[..., -1:]
        holding_stock = torch.zeros_like(prices_dis)  # t = 0 is meaningless
        for t in range(n_timestep - 1):
            info_dyn = holding_stock
            all_info_t = self.compute_info_t(info_dyn, info, t)
            holding_stock[:, t + 1, :] = self.compute_holding_stock_tplus1(all_info_t)
            wealth_incr = holding_stock[:, t + 1, :] * (
                prices_dis[:, t + 1, :] - prices_dis[:, t, :]
            )
            wealth_incr_sum = torch.sum(wealth_incr, dim=-1, keepdim=False)
            wealth_dis[t + 1] = wealth_dis[t] + wealth_incr_sum
        return wealth_dis

    def compute_info_t(self, info_dyn: Tensor, info: Tensor, t=None) -> Tensor:
        # all_info_t = torch.cat(
        #         [info[:, t, :], info_dyn[:,t,:]],
        #         dim=-1)
        all_info_t = info[:, t, :]
        return all_info_t

    def compute_holding_stock_tplus1(
        self, all_info_t: Tensor, t=None
    ) -> Tensor:  # We might use t here if it is deep hedge
        holding_stock_tplus1 = self.strategy(all_info_t)
        return holding_stock_tplus1

    def compute_pnl(
        self, prices: Tensor, info: Tensor, init_wealth: Tensor, payoff: Tensor
    ):
        wealth = self.forward(prices, info, init_wealth)
        terminal_wealth = wealth[:, -1]
        pnl = terminal_wealth - payoff
        return pnl

    def compute_loss(self, input: List[Tensor]):
        prices, info, payoff = input
        init_wealth = torch.tensor(0.0)
        pnl = self.compute_pnl(prices, info, init_wealth, payoff)
        loss = self.risk(pnl)
        return loss

    def pricer(self, input):
        prices, info, payoff = input
        with torch.no_grad():
            init_wealth = torch.tensor(0.0)
            pnl = self.compute_pnl(prices, info, init_wealth, payoff)
            price = self.risk.cash(pnl)
        return price


class EfficientHedger(Hedger):
    def __init__(
        self,
        strategy: torch.nn.Module,
        init_wealth: Tensor,
        risk: LossMeasure = EntropicRiskMeasure(),
        ad_bound=0.0,
    ):
        super().__init__(strategy, risk)
        self.init_wealth = init_wealth
        self.ad_bound = ad_bound

    def compute_loss(self, input: List[Tensor]):
        prices, info, payoff = input
        init_wealth = self.init_wealth
        wealth = self.forward(prices, info, init_wealth)
        terminal_wealth = wealth[:, -1]
        pnl = terminal_wealth - payoff
        ad_cost = admissible_cost(wealth, self.ad_bound)
        loss = self.risk(pnl) + ad_cost
        return loss
