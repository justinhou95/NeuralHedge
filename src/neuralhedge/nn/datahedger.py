from collections import defaultdict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from abc import ABC, abstractmethod

from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, proportional_cost, no_cost, admissible_cost, log_utility
from neuralhedge._utils.plotting import plot_pnl, plot_history
from neuralhedge.data.base import HedgerDataset

from os import path as pt

class HedgerBase(Module, ABC):
    @abstractmethod
    def add_wealth():
        pass
    @abstractmethod
    def compute_cost():
        pass
    @abstractmethod
    def compute_hedge():
        pass
    @abstractmethod
    def compute_loss():
        pass
    @abstractmethod
    def compute_info():
        pass

class Hedger(HedgerBase):

    """Hedger to hedge with only data generated but not the generating class
    Args:
        - model (torch.nn.Module) or models (List[Module]): depending on independent neural network at each time step or the same neural network at each time
        - dataset_market (Dataset)
        - criterion (HedgeLoss)
    """

    def __init__(
        self,
        model: Module,
        cost_functional = no_cost,
        ):
        super().__init__()
        self.model = model
        self.cost_functional = cost_functional
        self.history = defaultdict(list)
        self.steps = 1

    def forward(self, input: List[Tensor]):
        """Compute the terminal wealth

        Args:
            input = prices, information, payoff
            prices = (n_samples, n_step+1, n_asset)
            information = (n_samples, n_step+1, n_feature)

        Note:
            V_t: Wealth process 
            I_t: Information process = (information, state_information)
            H: hedging strategy functional
            H_t: holding process
            S_t: Price process
            C_t: Cost process
            dV_t = H(I_t)dS_t - dC_t
            
        Returns:
            V_T: torch.Tensor
        """
        prices, information, payoff = input 
        batch_size = prices.shape[0]
        wealth = [torch.zeros([batch_size]) for t in range(prices.shape[1])]
        holding = torch.zeros_like(prices)
        for t in range(prices.shape[1]-1):
            all_information = self.compute_info(holding, information, t)   # compute information at time t 
            holding[:,t+1,:] = self.compute_hedge(all_information, t)  # compute the holding at time t+1
            cost = self.compute_cost(holding, prices, t)  # compute the transaction cost 
            wealth[t+1] = wealth[t] + self.add_wealth(holding, prices, cost, t)
        return wealth

    def compute_info(self, holding: Tensor, info: Tensor, t = None) -> Tensor:
        state_info = holding[:,t,:]
        all_info = torch.cat(
                [info[:, t, :], state_info], 
                dim=-1)
        return all_info
    
    def compute_hedge(self, all_info: Tensor, t = None) -> Tensor:   # We might use t here if it is deep hedge
        holding = self.model(all_info)
        return holding   

    def compute_cost(self, holding, prices, t) -> Tensor:
        holding_diff = holding[:,t+1,:] - holding[:,t,:]
        price_now = prices[:,t,:]
        cost = self.cost_functional(holding_diff, price_now)
        return cost

    def add_wealth(self, holding, prices, cost, t):
        wealth_incr = holding[:,t+1,:] * (prices[:,t+1,:] - prices[:,t,:]) - cost
        wealth_incr = torch.sum(wealth_incr , dim=-1, keepdim=False)
        return wealth_incr
    
    def compute_pnl(self, input: List[Tensor]):
        prices, information, payoff = input
        wealth= self.forward(input)
        pnl = wealth[-1] - payoff
        return pnl

    def compute_loss(self, input: List[Tensor]):
        pnl = self.compute_pnl(input)
        return self.risk(pnl)
    
    def pricer(self, data):
        with torch.no_grad():
            pnl = self.compute_pnl(data)
            price = self.risk.cash(pnl)
        return price
        
    def fit(
        self, hedger_ds: Dataset,
        risk: LossMeasure = EntropicRiskMeasure(),
        EPOCHS=100, batch_size=256, 
        optimizer=torch.optim.Adam, 
        lr_scheduler_gamma = 0.9,
        lr=0.01,
        record_dir = None
        ):
        self.risk = risk
        hedger_dl = DataLoader(
            hedger_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        self.optimizer = optimizer(self.parameters(),lr = lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_scheduler_gamma)
            
        self.train(True)
        progress = tqdm(range(EPOCHS))
        for epoch in progress:
            for i, data in enumerate(hedger_dl):
                self.optimizer.zero_grad()
                loss = self.compute_loss(data)
                loss.backward()
                self.optimizer.step()
                self.history['loss'].append(loss.item())
                self.record_history()
                progress.desc = "Loss=" + str(loss.item())
                self.steps += 1
            lr_scheduler.step()
            if epoch % 10 == 0 and record_dir:
                self.record_parameter(record_dir)

    def record_history(self,):
        pass

    def record_parameter(self, record_dir):
        file_path = pt.join(record_dir,"parameter" +str(self.steps) + ".pth")
        torch.save(self.state_dict(), file_path)


