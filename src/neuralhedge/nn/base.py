from collections import defaultdict
from turtle import forward
from typing import List, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from neuralhedge.nn.loss import EntropicRiskMeasure, LossMeasure, proportional_cost, no_cost, admissible_cost, log_utility
from os import path as pt

class BaseModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_loss(self, input: List[Tensor]) -> Tensor:
        return torch.zeros(1)