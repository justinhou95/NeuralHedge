.. NeuralHedge documentation master file, created by
   sphinx-quickstart on Mon Aug  5 09:35:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to NeuralHedge's documentation!
=========================================
**NeuralHedge** is a Python library implements PyTorch realization of Hedging, Utility Maximization and Portfolio Optimization with neural networks. In particular it is made up of fully data-driven neural solver, where you could input your own data. The strategy, model and solver class are fully decoupled, making it very easy to implement your personalized problem. This implementation is very light weighted and essentially only relies on PyTorch for convenient future maintenance.


Installation📦
--------------------
Install the latest stable release:

.. code-block:: console

   (.venv) $ pip install neuralhedge

Install the latest github version:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/justinhou95/NeuralHedge.git

Clone the github repo for development to access to tests, tutorials and scripts.

.. code-block:: console

   (.venv) $ git clone https://github.com/justinhou95/NeuralHedge.git

and install in `develop mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_

.. code-block:: console

   (.venv) $ cd NeuralHedge

   (.venv) $ pip install -e .



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   examples
   api/modules

