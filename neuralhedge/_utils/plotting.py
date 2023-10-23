import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path as pt
from torch import Tensor 

def to_numpy(x: torch.Tensor): 
    return x.numpy()

def plot_hedge(hedger, data, init_wealth, record_dir = None):
    paths,_ , payoff = data
    payoff = payoff
    with torch.no_grad():
        terminal_wealth = hedger(data)[-1] + init_wealth
    pnl = terminal_wealth - payoff

    fig = plt.figure()
    plt.scatter(paths[:,-1,0], terminal_wealth)
    plt.scatter(paths[:,-1,0], payoff)
    plt.title(f"Hedge (inital:{init_wealth.numpy():.2f})")
    plt.xlabel("Stock")
    plt.ylabel("Wealth")
    plt.grid()
    if record_dir is not None:
        file_path = pt.join(record_dir, "plot_hedge_payoff.png")
        plt.savefig(file_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    plot_pnl(pnl, record_dir)


def plot_pnl(pnl, record_dir = None):
    fig = plt.figure()
    plt.hist(pnl, bins = list(np.linspace(-3,3,100)))
    plt.title("Profit-Loss Histograms")
    plt.xlabel("Profit-Loss")
    plt.ylabel("Number of events")
    plt.grid()
    if record_dir is not None:
        file_path = pt.join(record_dir, "plot_pnl_histogram.png")
        plt.savefig(file_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    
    
def plot_history(history, record_dir = None):
    smooth_history = np.convolve(np.array(history),np.ones(100)/100, 'valid')
    # smooth_history = history
    fig, ax = plt.subplots()
    plt.plot(smooth_history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss trajectory ")
    if record_dir is not None:
        file_path = pt.join(record_dir, "plot_train_loss.png")
        fig.savefig(file_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_data(data, plot_samples = 100):
    paths, info, payoff = data
    print('Shape of paths: ', paths.shape)
    print('Shape of information: ', info.shape)
    print('Shape of payoff: ', payoff.shape)
    
    plt.figure
    plt.plot(to_numpy(paths)[:plot_samples,:,0].T)
    plt.grid()
    plt.show()

    plt.figure
    plt.scatter(to_numpy(paths)[:,-1,0], to_numpy(payoff))
    plt.grid()
    plt.show()