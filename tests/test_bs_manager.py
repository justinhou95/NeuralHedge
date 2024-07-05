
import sys
sys.path.append(".")
import os

import torch
from os import makedirs

from neuralhedge.data.base import HedgerDataset
from neuralhedge.nn import datahedger, mlp, loss, blackschole
from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, SquareMeasure, ExpectedShortfall
from neuralhedge._utils.plotting import plot_pnl, plot_history, plot_data, plot_hedge
from neuralhedge.data.stochastic import BlackScholesDataset, simulate_time
from neuralhedge.nn.contigent import EuropeanVanilla

def experiment(hedge_ds: HedgerDataset, 
               record_dir, 
               risk_measure: LossMeasure):
    makedirs(record_dir, exist_ok=True)
    model = mlp.NeuralNetSequential(n_output = hedge_ds.paths.shape[-1])
    hedger = datahedger.Hedger(model) 
    hedger.fit(hedge_ds, EPOCHS=100, risk=risk_measure, record_dir=record_dir) 
    price = hedger.pricer(hedge_ds.data)
    plot_history(hedger.history, record_dir = record_dir)
    plot_hedge(hedger, hedge_ds.data, price, record_dir = record_dir)


def main():

    # Generate Black Scholes paths
    n_sample = 50000
    n_timestep = 30
    dt = 1/365
    ds_bs = BlackScholesDataset(n_sample = n_sample,
                            n_timestep = n_timestep,
                            dt = dt)
    paths = ds_bs.paths*100

    # Compute European call payoff 
    contigent = EuropeanVanilla(strike = 100., call = True)
    payoff = contigent.payoff(paths[:,-1,0])

    # Compute Information paths 
    info1 = torch.log(paths)
    info2 = simulate_time(n_sample, dt, n_timestep, reverse = True)
    info = torch.cat([info1,
                    info2],
                    dim = -1)
    
    data = [paths, info, payoff]
    hedge_ds = datahedger.HedgerDataset(data)
    
    # Test with Delta strategy
    model_delta = blackschole.BlackScholesDelta(sigma = ds_bs.sigma,
                                        risk_free_rate = 0.,
                                        strike = contigent.strike)
    bs_pricer = blackschole.BlackScholesPrice(sigma = ds_bs.sigma,
                                        risk_free_rate = 0.,
                                        strike = contigent.strike)
    bs_price = bs_pricer(info[0,0])[0]
    hedger = datahedger.Hedger(model_delta) 

    record_dir = './examples/numerical_results/bs_call_delta/'
    makedirs(record_dir, exist_ok=True)
    plot_hedge(hedger, hedge_ds.data, init_wealth=bs_price, record_dir=record_dir)



    record_dir = './examples/numerical_results/bs_call_entropic/'
    risk_measure = EntropicRiskMeasure()
    experiment(hedge_ds, record_dir, risk_measure)

    record_dir = './examples/numerical_results/bs_call_square/'
    risk_measure = SquareMeasure()
    experiment(hedge_ds, record_dir, risk_measure)

    record_dir = './examples/numerical_results/bs_call_es50'
    alpha = 0.5
    q = 1-alpha
    risk_measure = ExpectedShortfall(q)
    experiment(hedge_ds, record_dir, risk_measure)

    record_dir = './examples/numerical_results/bs_call_es99'
    alpha = 0.99
    q = 1-alpha
    risk_measure = ExpectedShortfall(q)
    experiment(hedge_ds, record_dir, risk_measure)


if __name__ == '__main__':
    main()



