from typing import List
import torch
from torch import Tensor


class BaseModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_loss(self, input: List[Tensor]) -> Tensor:
        return torch.zeros(1)
