from logging import Manager
from os import makedirs
from os import path as pt

import torch

from neuralhedge.data.base import HedgerDataset, ManagerDataset
from neuralhedge.nn import datahedger, loss, network
from neuralhedge.nn.loss import (
    EntropicRiskMeasure,
    ExpectedShortfall,
    PowerMeasure,
    SquareMeasure,
    log_utility,
)
from neuralhedge.utils.plotting import plot_hedge, plot_history, plot_pnl, plot_strategy


def evaluate_bs_deep_hedge(
    hedger: datahedger.Hedger, ds: HedgerDataset, bs_price, record_dir: str = ""
):
    if record_dir:
        print(f"Save evaluation at: {record_dir}")
        makedirs(record_dir, exist_ok=True)
        torch.save(hedger.strategy.state_dict(), pt.join(record_dir, "model.pth"))

    init_wealth = bs_price
    prices, info, payoff = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices, info, init_wealth)
        terminal_wealth = wealth[:, -1]
        pnl = terminal_wealth - payoff

    print(f"Entropic Measure: {EntropicRiskMeasure()(pnl):.2f}")
    print(f"Square Measure: {SquareMeasure()(pnl):.2f}")
    print(f"Power Measure (p=1): {PowerMeasure(p=1)(pnl):.2f}")
    print(f"Expected Shortfall (q=0.5): {ExpectedShortfall(q=0.5)(pnl):.2f}")
    print(f"Expected Shortfall (q=0.9): {ExpectedShortfall(q=0.9)(pnl):.2f}")

    plot_pnl(pnl, record_dir)
    plot_hedge(prices, terminal_wealth, payoff, init_wealth, record_dir)
    plot_strategy(hedger.strategy, ds, record_dir)


def evaluate_bs_efficient_hedge(
    hedger: datahedger.Hedger,
    ds: HedgerDataset,
    bs_price,
    init_wealth,
    record_dir: str = "",
):
    if record_dir:
        print(f"Save evaluation at: {record_dir}")
        makedirs(record_dir, exist_ok=True)
        torch.save(hedger.strategy.state_dict(), pt.join(record_dir, "model.pth"))

    print(f"BS price: {bs_price:.2f}")
    print(f"initial wealth: {init_wealth:.2f}")

    prices, info, payoff = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices, info, init_wealth)
        terminal_wealth = wealth[:, -1]
        pnl = terminal_wealth - payoff

    print(f"Risk(p=1): {loss.PowerMeasure(1)(pnl):.2f}")
    print(f"Risk(p=2): {loss.PowerMeasure(2)(pnl):.2f}")
    print(f"Risk(p=0.5): {loss.PowerMeasure(0.5)(pnl):.2f}")

    print(f"Entropic Measure: {EntropicRiskMeasure()(pnl):.2f}")
    print(f"Square Measure: {SquareMeasure()(pnl):.2f}")
    print(f"Expected Shortfall (q=0.5): {ExpectedShortfall(q=0.5)(pnl):.2f}")
    print(f"Expected Shortfall (q=0.9): {ExpectedShortfall(q=0.9)(pnl):.2f}")

    plot_pnl(pnl, record_dir)
    plot_hedge(prices, terminal_wealth, payoff, init_wealth, record_dir)
    plot_strategy(hedger.strategy, ds, record_dir)


from neuralhedge.nn.datamanager import Manager


def evaluate_merton(
    hedger: Manager,
    ds: ManagerDataset,
    init_wealth,
    record_dir: str = "",
):
    if record_dir:
        print(f"Save evaluation at: {record_dir}")
        makedirs(record_dir, exist_ok=True)
        torch.save(hedger.strategy.state_dict(), pt.join(record_dir, "model.pth"))

    print(f"initial wealth: {init_wealth:.2f}")

    prices, info = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices, info)
        terminal_wealth = wealth[:, -1] * init_wealth

    utility = log_utility(terminal_wealth).mean().numpy()
    print(f"Log utility: {utility:.2f}")

    plot_pnl(terminal_wealth, record_dir)


def evaluate_mv(
    hedger: Manager,
    ds: ManagerDataset,
):
    prices, info = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices, info)
        terminal_wealth = wealth[:, -1]
    mean = terminal_wealth.mean().item()
    var = terminal_wealth.var().item()
    return mean, var
