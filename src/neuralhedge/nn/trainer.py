from collections import defaultdict
from typing import List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neuralhedge.nn.base import BaseModel


class Trainer(torch.nn.Module):
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model
        self.history = defaultdict(list)

    def forward(self, input: List[Tensor]):
        loss = self.model.compute_loss(input)
        return loss

    def fit(
        self,
        hedger_ds: Dataset,
        EPOCHS=100,
        batch_size=256,
        optimizer=torch.optim.Adam,
        lr_scheduler_gamma=1.0,
        lr=0.01,
    ):
        self.steps = 1

        hedger_dl = DataLoader(
            hedger_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.optimizer = optimizer(self.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_scheduler_gamma
        )

        self.train(True)
        best_loss = 1e10
        progress = tqdm(range(EPOCHS))
        for epoch in progress:
            for i, data in enumerate(hedger_dl):
                self.optimizer.zero_grad()
                loss = self(data)
                self.history["loss"].append(loss.item())
                progress.desc = "Loss=" + str(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    self.best_weights = self.model.state_dict()
                loss.backward()
                self.optimizer.step()
                self.steps += 1
            if epoch % 100 == 0:
                lr_scheduler.step()
        self.model.load_state_dict(self.best_weights)
