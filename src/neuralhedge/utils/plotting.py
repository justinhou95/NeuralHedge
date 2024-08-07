from os import path as pt
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from h11 import Data
from torch import Tensor
from torch.utils.data import Dataset

import neuralhedge
from neuralhedge.data.base import HedgerDataset, ManagerDataset
from neuralhedge.nn.base import BaseModel
from neuralhedge.nn.loss import EntropicRiskMeasure, SquareMeasure


def to_numpy(x: torch.Tensor):
    return x.numpy()


def plot_hedge(prices, terminal_wealth, payoff, init_wealth, record_dir=None):
    r"""
    Plot terminal prices v.s. terminal wealth and terminal prices v.s. payoff
    """

    fig = plt.figure()
    plt.scatter(prices[:, -1, 0], terminal_wealth)
    plt.scatter(prices[:, -1, 0], payoff)

    plt.title(f"Hedge (inital:{init_wealth.numpy():.2f})")
    plt.xlabel("Stock")
    plt.ylabel("Wealth")
    plt.grid()

    if record_dir:
        file_path = pt.join(record_dir, "plot_hedge_payoff.png")
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close(fig)


def plot_pnl(pnl, record_dir=None):
    r"""
    Plot PnL histogram
    """
    fig = plt.figure()
    plt.hist(pnl, bins=100)
    plt.title("Profit-Loss Histograms")
    plt.xlabel("Profit-Loss")
    plt.ylabel("Number of events")
    plt.grid()
    if record_dir:
        file_path = pt.join(record_dir, "plot_pnl_histogram.png")
        plt.savefig(file_path)
    else:
        plt.show()
    plt.close(fig)


def plot_history(history: dict, record_dir=None):
    r"""
    Plot training history
    Arguments:
        history (:class:`Dict`): loss is one key of history
    """

    smooth_history = np.convolve(np.array(history["loss"]), np.ones(100) / 100, "valid")
    # smooth_history = history
    fig, ax = plt.subplots()
    plt.plot(smooth_history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss trajectory ")
    if record_dir:
        file_path = pt.join(record_dir, "plot_train_loss.png")
        fig.savefig(file_path)
    else:
        plt.show()
    plt.close(fig)


def plot_hedge_ds(ds: HedgerDataset, plot_samples=100):
    r"""
    Print shape of paths, info and payoff. Plot paths and plot payoff
    """
    paths, info, payoff = ds.data
    print("Shape of paths: ", paths.shape)
    print("Shape of information: ", info.shape)
    print("Shape of payoff: ", payoff.shape)

    plt.figure
    plt.plot(to_numpy(paths)[:plot_samples, :, 0].T)
    plt.plot(to_numpy(paths)[0, :, -1], label="risk-free", c="k")
    plt.title("prices paths")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure
    plt.scatter(to_numpy(paths)[:, -1, 0], to_numpy(payoff))
    plt.title("payoff")
    plt.grid()
    plt.show()


def plot_manage_ds(ds: ManagerDataset, plot_samples=100):
    r"""
    Print shape of paths and info. Plot paths
    """
    paths, info = ds.data
    print("Shape of paths: ", paths.shape)
    print("Shape of information: ", info.shape)

    plt.figure
    plt.plot(to_numpy(paths)[:plot_samples, :, 0].T)
    plt.plot(to_numpy(paths)[0, :, -1], label="risk-free", c="k")
    plt.title("prices paths")
    plt.legend()
    plt.grid()
    plt.show()


def plot_strategy(strategy: torch.nn.Module, ds_test: HedgerDataset, record_dir=""):
    r"""
    Plot strategy at different timestamps
    """

    observe_window = 3
    n_timestep = ds_test.prices.shape[1]
    observe_time = np.unique(np.arange(n_timestep) // observe_window) * observe_window

    fig, axes = plt.subplots(
        3,
        len(observe_time) // 3 + 1,
        figsize=[12, 5],
        sharex=True,
        sharey=True,
    )
    for i, t in enumerate(observe_time):
        ax = axes.flatten()[i]  # type: ignore
        with torch.no_grad():
            x = ds_test.prices[:, t, 0]
            y = strategy(ds_test.info[:, t, :])[..., 0]
        ax.set_title(f"t = {t:.2f}")
        ax.scatter(x, y, s=1)
        ax.set_xticklabels([])

    if record_dir:
        file_path = pt.join(record_dir, "strategy.png")
        fig.savefig(file_path)
    else:
        plt.show()
    plt.close(fig)
