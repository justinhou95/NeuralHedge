# neuralhedge 📈
This library implements PyTorch realization of Hedging, Utility Maximization and Portfolio Optimization with neural networks. In particular it is made up of fully data-driven neural solver, where you could input your own data. The strategy, model and solver class are fully decoupled, making it very easy to implement your personalized problem. This implementation is very light weighted and essentially only relies on PyTorch for convenient future maintenance.


## Installation 📦

Install the latest stable release:

```bash
$ pip install neuralhedge
``` 

Install the latest github version:

```bash
$ pip install git+https://github.com/justinhou95/NeuralHedge.git
``` 

Clone the github repo for development to access to tests, tutorials and scripts.
```bash
$ git clone https://github.com/justinhou95/NeuralHedge.git
```
and install in [develop mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
```bash
$ cd NeuralHedge
$ pip install -e .
``` 

## Tutorials 📚
To help you to understand how easy to train your models with neuralhedge, we also provide tutorials: 

- [deephedge.ipynb](https://github.com/justinhou95/NeuralHedge/blob/main/notebooks/deephedge.ipynb) shows you how to hedge an European call option under Black Scholes model with neural network.
- [efficienthedge.ipynb](https://github.com/justinhou95/NeuralHedge/blob/main/notebooks/efficienthedge.ipynb) shows you how to efficent (partial) hedge an European call option with insufficent endowment with neural network.
- [logutiliy.ipynb](https://github.com/justinhou95/NeuralHedge/blob/main/notebooks/logutility.ipynb) shows you how to solve Merton's portfolio problem with neural network.
- [mean_variance.ipynb](https://github.com/justinhou95/NeuralHedge/blob/main/notebooks/mean_variance.ipynb) shows you how to solve the mean-variance portfolio problem.


## Dealing with issues 🛠️

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/justinhou95/NeuralHedge/issues).


## Contributing 🚀
You want to contribute to this library, that will be cool!