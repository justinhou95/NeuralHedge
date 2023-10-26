
import sys
sys.path.append(".")

import torch
from os import makedirs

from neuralhedge.data.base import HedgerDataset
from neuralhedge.nn import datahedger, mlp, loss, blackschole
from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, SquareMeasure, ExpectedShortfall, admissible_cost
from neuralhedge._utils.plotting import plot_pnl, plot_history, plot_data, plot_hedge
from neuralhedge.data.stochastic import BlackScholesDataset, simulate_time
from neuralhedge.nn.contigent import EuropeanVanilla
from neuralhedge.nn.efficienthedger import EfficientHedger



def experiment(hedge_ds: HedgerDataset, 
               record_dir, 
               initial_price,
               admissible_bound,
               p):
    print('='*20)
    print('p = ', p)
    print('='*20)

    makedirs(record_dir, exist_ok=True)
    model = mlp.NeuralNetSequential(n_output = hedge_ds.paths.shape[-1])
    hedger = EfficientHedger(model, initial_price, admissible_bound) 
    hedger.fit(hedge_ds, EPOCHS=100, risk=loss.PowerMeasure(p), record_dir=record_dir)  
    price = torch.tensor(initial_price)
    plot_history(hedger.history, record_dir = record_dir)
    plot_hedge(hedger, hedge_ds.data, price, record_dir = record_dir)
    with torch.no_grad():
        wealth = hedger(hedge_ds.data)
        avg_ad = admissible_cost(wealth).numpy()
    print('Average admissibility break: {:.2f}'.format(avg_ad))


def main():

    # Generate Black Scholes paths
    n_sample = 50000
    n_timestep = 30*5
    dt = 1/365/5

    ds_bs = BlackScholesDataset(n_sample = n_sample,
                            n_timestep = n_timestep,
                            dt = dt)
    paths = ds_bs.paths*100


    contigent = EuropeanVanilla(strike = 100., call = True)
    payoff = contigent.payoff(paths[:,-1,0])

    info1 = torch.log(paths)
    info2 = simulate_time(n_sample, dt, n_timestep, reverse = True)
    info = torch.cat([info1,
                    info2],
                    dim = -1)
    data = [paths, info, payoff]
    hedge_ds = datahedger.HedgerDataset(data)


    initial_price = 2.0
    admissible_bound = -5

    for p in [1, 2, 0.5]:
        record_dir = './examples/numerical_results/bs_call_efficient_' + str(p) + '/'
        experiment(hedge_ds, record_dir, initial_price, admissible_bound,p)
        


if __name__ == '__main__':
    main()



