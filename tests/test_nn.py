from neuralhedge import nn
from neuralhedge.nn import network


def test_network():
    strategy = network.NeuralNetSequential(1, 1, 1)
