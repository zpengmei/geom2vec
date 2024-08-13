from typing import Optional

import numpy as np
import torch
from adam_atan2_pytorch import AdamAtan2
from grokfast_pytorch import GrokFastAdamW
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class VCN(nn.Module):
    r"""
    Variational committor network for estimating the committor function.
    """

    def __init__(
        self,
        lobe: nn.Module,
        *,
        optimizer: str = "adam",
        device: str = "cuda",
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        epsilon: float = 1e-1,
        k: float = 10.0,
        lag_time: float = 1.0,
        save_model_interval: Optional[int] = None,
    ):
        super().__init__()

        self._lobe = lobe
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._device = torch.device(device)
        self._epsilon = epsilon
        self._k = k
        self._lag_time = lag_time

        optimizer_types = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
            "grokfastadamw": GrokFastAdamW,
            "adamatan2": AdamAtan2,
        }
        self._optimizer = optimizer_types[optimizer](
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self._step = 0
        self._save_model_interval = save_model_interval
        self._save_models = []
        self._training_scores = []
        self._validation_scores = []
        self._training_bc_losses = []
        self._validation_bc_losses = []

    @property
    def training_scores(self):
        return np.array(self._training_scores)

    @property
    def validation_scores(self):
        return np.array(self._validation_scores)

    @property
    def training_bc_losses(self):
        return np.array(self._training_bc_losses)

    @property
    def validation_bc_losses(self):
        return np.array(self._validation_bc_losses)

    def _loss_fns(self, batch):
        model = self._lobe
        device = self._device
        eps = self._epsilon
        k = self._k
        lag_time = self._lag_time

        batch = [tensor.to(device) for tensor in batch]
        x0, x1, a0, a1, b0, b1 = batch

        x = torch.cat([x0, x1])
        a = torch.cat([a0, a1])
        b = torch.cat([b0, b1])

        u = model(x)
        q = torch.where(a, 0, torch.where(b, 1, torch.clamp(u, 0, 1)))
        bc = a * (u - (0 - eps)) ** 2 + b * (u - (1 + eps)) ** 2

        q0, q1 = torch.unflatten(q, 0, (2, -1))
        bc0, bc1 = torch.unflatten(bc, 0, (2, -1))

        score = (q0 - q1) ** 2 / (2 * lag_time)
        bc_loss = 0.5 * k * (bc0 + bc1)

        return score, bc_loss

    def fit(
        self,
        train_loader: DataLoader,
        *,
        n_epochs: int = 1,
        validation_loader: Optional[DataLoader] = None,
        progress=tqdm,
        train_patience: int = 1000,
        valid_patience: int = 1000,
        train_valid_interval: int = 1000,
    ):
        r"""
        Fit the committor network to the training data.

        Parameters
        ----------
        train_loader: training data loader. Should yield batches of time-lagged
            data with shape (2, n_batch, n_dim), in_A, and in_B
        n_epochs: number of epochs (passes through the data set)
        validation_loader: validation data loader
        progress
        train_patience: number of steps to wait for training loss to improve
        valid_patience: number of steps to wait for validation loss to improve

        Returns
        -------

        """
        self._step = 0

        best_train_score = torch.inf
        best_valid_score = torch.inf
        train_patience_counter = 0
        valid_patience_counter = 0
        best_lobe_state = self._lobe.state_dict()

        for epoch in progress(
            range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            for batch in progress(
                train_loader, desc="batch", total=len(train_loader), leave=False
            ):
                self._step += 1

                score, bc_loss = self._training_step(batch)
                loss = score + bc_loss

                # early stopping on training loss
                train_patience_counter += 1
                if loss < best_train_score:
                    best_train_score = loss
                    train_patience_counter = 0
                if train_patience_counter >= train_patience:
                    print(f"Training patience reached at epoch {epoch}")
                    self._lobe.load_state_dict(best_lobe_state)
                    return self

                if (
                    validation_loader is not None
                    and self._step % train_valid_interval == 0
                ):
                    score, bc_loss = self._validation_loop(
                        validation_loader, progress=progress
                    )

                    # early stopping on validation score
                    valid_patience_counter += 1
                    if score < best_valid_score:
                        best_valid_score = score
                        valid_patience_counter = 0
                        best_lobe_state = self._lobe.state_dict()
                    if valid_patience_counter >= valid_patience:
                        print(f"Validation patience reached at epoch {epoch}")
                        self._lobe.load_state_dict(best_lobe_state)
                        return self

                    if self._save_model_interval is not None:
                        if (epoch + 1) % self._save_model_interval == 0:
                            # save the model with the epoch and the losses
                            self._save_models.append(
                                (epoch, score, bc_loss, self._lobe.state_dict())
                            )

        self._lobe.load_state_dict(best_lobe_state)
        return self

    def _training_step(self, batch):
        self._lobe.train()
        self._optimizer.zero_grad()
        score, bc_loss = self._loss_fns(batch)
        score = torch.mean(score)
        bc_loss = torch.mean(bc_loss)
        loss = score + bc_loss
        loss.backward()
        self._optimizer.step()
        self._training_scores.append(score.item())
        self._training_bc_losses.append(bc_loss.item())
        return score.item(), bc_loss.item()

    def _validation_loop(self, validation_loader, progress=tqdm):
        self._lobe.eval()
        with torch.no_grad():
            score = 0.0
            bc_loss = 0.0
            for batch in progress(
                validation_loader,
                desc="validation",
                total=len(validation_loader),
                leave=False,
            ):
                score_batch, bc_loss_batch = self._loss_fns(batch)
                score += torch.sum(score_batch).item()
                bc_loss += torch.sum(bc_loss_batch).item()
            score /= len(validation_loader)
            bc_loss /= len(validation_loader)
            self._validation_scores.append(score)
            self._validation_bc_losses.append(bc_loss)
        return score, bc_loss

    def transform(self, dataset, batch_size):
        model = self._lobe
        device = self._device

        model.eval()
        out_list = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
                x, a, b = [tensor.to(device) for tensor in batch]
                u = model(x)
                q = torch.where(a, 0, torch.where(b, 1, torch.clamp(u, 0, 1)))
                out_list.append(q.cpu())
        return torch.cat(out_list).numpy()

    def fetch_model(self):
        from copy import deepcopy

        return deepcopy(self._lobe)

    def save_model(self, path, name="lobe.pt"):
        import os

        torch.save(self._lobe.state_dict(), os.path.join(path, name))
        torch.save(self, os.path.join(path, "vcn.pt"))

        return self._lobe
