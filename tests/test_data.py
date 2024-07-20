import torch
from neuralhedge.data.base import HedgerDataset, ManagerDataset
from neuralhedge.data.market import BS_Market

torch.manual_seed(0)


def test_bs_market_hedge():
    bs_market = BS_Market()
    ds = bs_market.get_hedge_ds()
    assert isinstance(ds, HedgerDataset)
    assert (ds.prices, ds.info, ds.payoff) == ds.data
    assert len(ds.prices.shape) == 3
    assert len(ds.info.shape) == 3
    assert len(ds.payoff.shape) == 1
    assert len(ds.prices) == len(ds.info) == len(ds.payoff)


def test_bs_market_manage():
    bs_market = BS_Market()
    ds = bs_market.get_manage_ds()
    assert isinstance(ds, ManagerDataset)
    assert (ds.prices, ds.info) == ds.data
    assert len(ds.prices.shape) == 3
    assert len(ds.info.shape) == 3
    assert len(ds.prices) == len(ds.info)
