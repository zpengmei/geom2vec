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
        save_model_interval: Optional[int] = None,
    ):
        super().__init__()

        self._lobe = lobe
        self._learning_rate = learning_rate
        self._device = torch.device(device)
        self._epsilon = epsilon
        self._k = k

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

    def _variational_loss(self, x, ina, inb):
        r"""
        Variational loss function for the committor network.
        Parameters
        ----------
        x: torch.Tensor
            The network output, of dimensions (2, n_batch, 1), at time 0 and time 1
        ina: torch.Tensor
            The input data for the first state.
        inb: torch.Tensor
            The input data for the second state.

        Returns
        -------

        """

        k = self._k
        eps = self._epsilon
        q = torch.clip(x, min=0, max=1)
        q = torch.where(ina, 0, torch.where(inb, 1, q))
        q0, q1 = q
        loss_var = torch.mean(0.5 * (q0 - q1) ** 2)
        loss_boundary = torch.mean(0.5 * k * ina * (x - (0 - eps)) ** 2)
        loss_boundary += torch.mean(0.5 * k * inb * (x - (1 + eps)) ** 2)
        return loss_var, loss_boundary

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
            optimizer = self._optimizer
            model = self._lobe

            for data, ina, inb in progress(
                train_loader, desc="batch", total=len(train_loader), leave=False
            ):
                self._step += 1

                model.train()

                optimizer.zero_grad()
                data = data.to(self._device)
                ina = ina.to(self._device)
                inb = inb.to(self._device)

                x = model(data)
                loss_var, loss_boundary = self._variational_loss(x, ina, inb)
                loss = loss_var + loss_boundary
                loss.backward()
                optimizer.step()

                self._training_scores.append(loss_var.item())
                self._training_bc_losses.append(loss_boundary.item())

                # early stopping on training loss
                train_patience_counter += 1
                if loss.item() < best_train_score:
                    best_train_score = loss.item()
                    train_patience_counter = 0
                if train_patience_counter >= train_patience:
                    print(f"Training patience reached at epoch {epoch}")
                    self._lobe.load_state_dict(best_lobe_state)
                    return self

                if (
                    validation_loader is not None
                    and self._step % train_valid_interval == 0
                ):
                    with torch.no_grad():
                        model.eval()

                        scores = []
                        bc_losses = []
                        for data, ina, inb in progress(
                            validation_loader,
                            desc="validation",
                            total=len(validation_loader),
                            leave=False,
                        ):
                            data = data.to(self._device)
                            ina = ina.to(self._device)
                            inb = inb.to(self._device)
                            q = model(data)
                            loss_var, loss_boundary = self._variational_loss(
                                q, ina, inb
                            )
                            scores.append(loss_var.item())
                            bc_losses.append(loss_boundary.item())
                        mean_score = np.mean(scores)
                        mean_bc_loss = np.mean(bc_losses)
                        self._validation_scores.append(mean_score)
                        self._validation_bc_losses.append(mean_bc_loss)

                        # early stopping on validation score
                        valid_patience_counter += 1
                        if mean_score < best_valid_score:
                            best_valid_score = mean_score
                            valid_patience_counter = 0
                            best_lobe_state = self._lobe.state_dict()
                        if valid_patience_counter >= valid_patience:
                            print(f"Validation patience reached at epoch {epoch}")
                            self._lobe.load_state_dict(best_lobe_state)
                            return self

                    if self._save_model_interval is not None:
                        if (epoch + 1) % self._save_model_interval == 0:
                            # save the model with the epoch and the mean score
                            self._save_models.append(
                                (epoch, mean_score, mean_bc_loss, model.state_dict())
                            )

        self._lobe.load_state_dict(best_lobe_state)
        return self

    def transform(self, dataset, batch_size):
        model = self._lobe
        model.eval()
        device = self._device

        out_list = []
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for data, ina, inb in tqdm(loader):
                data = data.to(device)
                ina = ina.to(device)
                inb = inb.to(device)
                out = model(data)
                out = torch.clip(out, min=0, max=1)
                out = torch.where(ina, 0, torch.where(inb, 1, out))
                out_list.append(out.clone().detach().cpu())

        q = torch.cat(out_list, dim=0)
        q = q.numpy()
        return q

    def fetch_model(self):
        from copy import deepcopy

        return deepcopy(self._lobe)

    def save_model(self, path, name="lobe.pt"):
        import os

        torch.save(self._lobe.state_dict(), os.path.join(path, name))
        torch.save(self, os.path.join(path, "vcn.pt"))

        return self._lobe
