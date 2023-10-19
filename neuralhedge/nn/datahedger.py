from typing import List
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore

from abc import ABC, abstractmethod

from .loss import EntropicRiskMeasure, proportional_cost, no_cost
from neuralhedge._utils.plotting import plot_pnl, plot_history


class MarketDataset(Dataset):
    """Market information dataset.
    Args:
        - data_set (List[Tensor]): [prices, information, payoff]
    Shape:
        - prices: (n_samples, n_steps+1, n_assets)
        - information: (n_samples, >= n_steps , n_features)
        - payoff: (n_samples, 1)
    """

    def __init__(self, data: List[Tensor]):
        self.data = data
        self.prices, self.information, self.payoff = data

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, idx: int):
        return [self.prices[idx], self.information[idx], self.payoff[idx]]

class HedgerBase(Module, ABC):

    @abstractmethod
    def update_wealth():
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
    def compute_pnl():
        pass
    @abstractmethod
    def compute_all_information():
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

    def forward(self, input: List[Tensor]) -> Tensor:
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
        wealth = torch.zeros_like(prices)
        holding = torch.zeros_like(prices)
        for t in range(prices.shape[1]-1):
            all_information = self.compute_info(holding, information, t)   # compute information at time t 
            holding[:,t+1,:] = self.compute_hedge(all_information, t)  # compute the holding at time t+1
            cost = self.compute_cost(holding, prices, t)  # compute the transaction cost 
            wealth[:,t+1,:] = wealth[:,t,:] + self.update_wealth(holding, prices, cost, t)

        wealth = torch.sum(wealth, dim=-1, keepdim=True)
        return wealth

    def compute_info(self, holding: Tensor, information: Tensor, t = None) -> Tensor:
        state_information = holding[:,t,:]
        all_information = torch.cat(
                [information[:, t, :], state_information], 
                dim=-1)
        return all_information 

    def compute_hedge(self, all_information: Tensor, t = None) -> Tensor:   # We might use t here if it is deep hedge
        holding = self.model(all_information)
        return holding   

    def compute_cost(self, holding, prices, t) -> Tensor:
        holding_diff = holding[:,t+1,:] - holding[:,t,:]
        price_now = prices[:,t,:]
        cost = self.cost_functional(holding_diff, price_now)
        return cost

    def update_wealth(self, holding, prices, cost, t):
        wealth_incr = holding[:,t+1,:] * (prices[:,t+1,:] - prices[:,t,:]) - cost
        return wealth_incr
    
    # not used in forward
    def compute_pnl(self, input: List[Tensor]):
        prices, information, payoff = input
        wealth= self.forward(input)
        pnl = wealth - payoff
        return pnl

    def compute_loss(self, input: List[Tensor]):
        pnl = self.compute_pnl(input)
        return self.criterion(pnl)
        # TODO: make it clean herex
        

    def fit(
        self, dataset_market: MarketDataset,
        criterion: Module = EntropicRiskMeasure(),
        EPOCHS=100, batch_size=256, 
        optimizer=torch.optim.Adam, 
        lr=0.001,
        ):

        self.dataset_market = dataset_market
        self.n_paths = int(self.dataset_market.prices.shape[0]) 
        self.n_steps = int(self.dataset_market.prices.shape[1]) - 1  
        self.criterion = criterion
        

        self.dataloader_market = DataLoader(
            self.dataset_market, batch_size=batch_size, shuffle=True, num_workers=0)

        self.optimizer = optimizer(self.parameters(),lr = lr)
        self.history = []
        progress = tqdm(range(EPOCHS))
        for _ in progress:
            self.train(True)
            for i, data in enumerate(self.dataloader_market):
                self.optimizer.zero_grad()
                loss = self.compute_loss(data)
                loss.backward()
                self.optimizer.step()
                self.history.append(loss.item())
                progress.desc = "Loss=" + str(loss.item())

        return self.history

    # def pricer(self, data) -> Tensor:
    #     self.pnl = self.compute_pnl(data)
    #     self.price = self.criterion.cash(self.pnl)
    #     return self.price
        


# class DeepHedger(Hedger):
#     def __init__(self,models: List,):
#         super().__init__(None)
#         self.models = ModuleList(models)

#     def compute_hedge(self, all_information: Tensor, t = None) -> Tensor:
#         holding = self.models[t](all_information)
#         return holding
        
#     def compute_all_information(self, holding: Tensor, information: Tensor, t = None) -> Tensor:
#         return information[:, t, :]