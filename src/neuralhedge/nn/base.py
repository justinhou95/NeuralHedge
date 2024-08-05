from typing import List
import torch
from torch import Tensor


class BaseModel(torch.nn.Module):
    r"""
    Base class for models
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def compute_loss(self, input: List[Tensor]) -> Tensor:
        r"""
        Output loss to trainer
        Returns:
            loss (:class:`torch.Tensor`)

        """
        return torch.zeros(1)
