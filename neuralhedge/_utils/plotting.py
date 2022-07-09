import torch
import numpy as np
import matplotlib.pyplot as plt

# def to_numpy(tensor: torch.Tensor) -> np.array:
#     return tensor.cpu().detach().numpy()

def plot_hedge(hedger, data_set, pnl = True):
    paths,_,payoff = data_set
    wealth0 = hedger(data_set).detach().numpy()
    price = hedger.pricer(data_set).detach().numpy()
    wealth = wealth0 + price
    

    plt.figure
    plt.scatter(paths[:,-1,0], wealth[:,-1,0])
    plt.scatter(paths[:,-1,0], payoff[:,-1,0])
    plt.title("Hedge")
    plt.xlabel("Prices")
    plt.ylabel("Wealth")
    plt.grid()
    plt.show()

    print('Price: ', price)

    if pnl:
        pnl = hedger.compute_pnl(data_set).detach().numpy()
        plot_pnl(pnl[:,-1,:])


def plot_pnl(pnl):
    plt.figure()
    plt.hist(pnl, bins=100)
    plt.title("Profit-Loss Histograms")
    plt.xlabel("Profit-Loss")
    plt.ylabel("Number of events")
    plt.grid()
    plt.show()
    
def plot_history(history):
    smooth_history = np.convolve(np.array(history),np.ones(100)/100, 'valid')
    # smooth_history = history
    plt.plot(smooth_history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss trajectory ")
    plt.show()


def plot_data_set(data_set, plot_samples = 100):
    paths, information, payoff = data_set 
    print('Shape of paths: ', paths.shape)
    print('Shape of information: ', information.shape)
    print('Shape of payoff: ', payoff.shape)
    
    plt.figure
    plt.plot(paths[:plot_samples,:,0].T)
    plt.grid()
    plt.show()

    plt.figure
    plt.scatter(paths[:,-1,0], payoff[:,-1,0])
    plt.grid()
    plt.show()