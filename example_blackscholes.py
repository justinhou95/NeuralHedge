# Example to use Black-Scholes' delta-hedging strategy as a hedging model
import numpy as np
import torch

from math import ceil
import matplotlib.pyplot as plt

from neuralhedge.nn import datahedger, contigent, mlp, loss, blackschole
from neuralhedge.nn.loss import ExpectedShortfall
from neuralhedge.market import stochastic, markets
from neuralhedge._utils.plotting import plot_pnl, plot_history, plot_data_set, plot_hedge
from importlib import reload



if __name__ == "__main__":

    data_set, dataset_market = markets.BlackScholesTestMarket()
    paths, information, payoff = data_set
    n_asset = paths.shape[-1]
    plot_data_set(data_set)
    true_price = payoff[:,-1].mean().numpy()
    print('True price is: ', payoff[:,-1].mean())

    # Delta hedge 
    plt.plot('Delta Hedge')
    model = blackschole.BlackScholesDelta(sigma=0.2,strike=100.)
    hedger = datahedger.Hedger(model) 
    plot_hedge(hedger, data_set, price = true_price)

    
    # Markov hedge
    plt.plot('Markov Hedge')
    model = mlp.NeuralNetSequential(n_output = n_asset)
    hedger = datahedger.Hedger(model) 
    history = hedger.fit(dataset_market, EPOCHS=80) 
    plot_history(history)
    plot_hedge(hedger, data_set)

    # Markov hedge + previous hedge
    plt.plot('Markov Hedge + previous hedge')
    model = mlp.NeuralNetSequential(n_output = n_asset)
    hedger = datahedger.Hedger(model) 
    history = hedger.fit(dataset_market, EPOCHS=80) 
    plot_history(history)
    plot_hedge(hedger, data_set)

    # Deep hedge 
    plt.plot('Deep Hedge')
    model = mlp.NeuralNetSequential(n_output = n_asset)
    hedger = datahedger.Hedger(model) 
    history = hedger.fit(dataset_market, EPOCHS=80) 
    plot_history(history)
    plot_hedge(hedger, data_set)
