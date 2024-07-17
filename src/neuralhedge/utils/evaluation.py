import torch
from os import makedirs
from os import path as pt
from neuralhedge.data.base import HedgerDataset
from neuralhedge.nn import datahedger, network
from neuralhedge.nn import loss
from neuralhedge.utils.plotting import plot_pnl, plot_history, plot_hedge
from neuralhedge.nn.loss import EntropicRiskMeasure, SquareMeasure


def evaluate_bs_deep_hedge(hedger : datahedger.Hedger, 
                           ds: HedgerDataset, 
                           bs_price, 
                           record_dir: str = ''):
    if record_dir:
        print(f'Save evaluation at: {record_dir}')
        makedirs(record_dir, exist_ok=True)

    torch.save(hedger.strategy.state_dict(), pt.join(record_dir,'model.pth'))
    
    hedger_price = hedger.pricer(ds.data)
    print(f'{hedger.risk} Hedger price: {hedger_price:.2f}')
    print(f'BS price: {bs_price:.2f}')
    init_wealth = bs_price

    prices, info, payoff = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices,info,init_wealth)
        terminal_wealth = wealth[:,-1]
        pnl = terminal_wealth - payoff

    print(f'Square Measure: {SquareMeasure()(pnl):.2f}')
    print(f'EntropicRiskMeasure: {EntropicRiskMeasure()(pnl):.2f}')
    plot_pnl(pnl, record_dir)
    plot_hedge(prices, terminal_wealth, payoff, init_wealth, record_dir)


def evaluate_bs_efficient_hedge(hedger : datahedger.Hedger, 
                           ds: HedgerDataset, 
                           bs_price, 
                           init_wealth,
                           record_dir: str = ''):
    if record_dir:
        print(f'Save evaluation at: {record_dir}')
        makedirs(record_dir, exist_ok=True)

    torch.save(hedger.strategy.state_dict(), pt.join(record_dir,'model.pth'))
    
    print(f'BS price: {bs_price:.2f}')
    print(f'BS price: {init_wealth:.2f}')

    prices, info, payoff = ds.data
    with torch.no_grad():
        wealth = hedger.forward(prices,info,init_wealth)
        terminal_wealth = wealth[:,-1]
        pnl = terminal_wealth - payoff

    print(f'Risk(p=1): {loss.PowerMeasure(1)(pnl):.2f}')
    print(f'Risk(p=2): {loss.PowerMeasure(2)(pnl):.2f}')
    print(f'Risk(p=0.5): {loss.PowerMeasure(0.5)(pnl):.2f}')


    plot_pnl(pnl, record_dir)
    plot_hedge(prices, terminal_wealth, payoff, init_wealth, record_dir)
