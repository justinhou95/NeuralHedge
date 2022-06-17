import torch
import numpy as np
import matplotlib.pyplot as plt

# def to_numpy(tensor: torch.Tensor) -> np.array:
#     return tensor.cpu().detach().numpy()

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