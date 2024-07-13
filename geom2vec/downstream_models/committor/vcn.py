from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit
from torch.utils.data import DataLoader
from tqdm import tqdm
from grokfast_pytorch import GrokFastAdamW


class VCN(nn.Module):
    r"""
    Variational committor network for estimating the committor function.
    """

    def __init__(
        self,
        lobe: nn.Module,
        optimizer: str = "adam",
        device: str = "cuda",
        learning_rate: float = 5e-4,
        weight_decay: float = 0,
        epsilon: float = 1e-1,
        k: float = 10,
        save_model_interval: Optional[int] = None,
    ):
        super(VCN, self).__init__()

        self._lobe = lobe
        self._learning_rate = learning_rate
        self._device = torch.device(device)
        self._epsilon = epsilon
        self._k = k

        assert optimizer in ["adam", "adamw", "sgd", "grokfastadamw"]

        if optimizer == "adam":
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adamw":
            self._optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self._optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "grokfastadamw":
            self._optimizer = GrokFastAdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self._step = 0
        self._save_model_interval = save_model_interval
        self._save_models = []
        self._training_scores = []
        self._validation_scores = []

    @property
    def training_scores(self):
        return np.array(self._training_scores)

    @property
    def validation_scores(self):
        return np.array(self._validation_scores)

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
        q = nn.sigmoid(x)

        q = q * (1 + 2 * eps) - eps
        q = torch.clip(q, min=0, max=1)
        q0, q1 = q
        loss_var = torch.mean(0.5 * (q0 - q1) ** 2)
        loss_boundary = torch.mean(0.5 * k * (x * ina + eps) ** 2)
        loss_boundary += torch.mean(0.5 * k * (x * inb - (1 + eps)) ** 2)
        return loss_var, loss_boundary

    def fit(
        self,
        train_loader: DataLoader,
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
        best_train_score = 0
        best_valid_score = 0
        train_patience_counter = 0
        valid_patience_counter = 0
        step_counter = 0
        best_lobe_state = self._lobe.state_dict()


        for epoch in progress(
            range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            optimizer = self._optimizer
            model = self._lobe
            k = self._k

            model.train()


            for data, ina, inb in progress(
                train_loader, desc="batch", total=len(train_loader), leave=False
            ):
                step_counter += 1
                optimizer.zero_grad()
                data = data.to(self._device)
                ina = ina.to(self._device)
                inb = inb.to(self._device)

                x = model(data)
                loss_var, loss_boundary = self._variational_loss(x, ina, inb)
                loss = loss_var + loss_boundary
                loss.backward()
                optimizer.step()

                self._training_scores.append(loss_var.item() + loss_boundary.item())

                if loss.item() < best_train_score:
                    best_train_score = loss.item()
                    train_patience_counter = 0
                else:
                    train_patience_counter += 1
                    if train_patience_counter >= train_patience:
                        print(f"Training patience reached at epoch {epoch}")
                        self._lobe.load_state_dict(best_lobe_state)
                        return self

                if validation_loader is not None and step_counter % train_valid_interval == 0:
                    with torch.no_grad():
                        model.eval()

                        losses = []
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
                            loss_var, loss_boundary = self._variational_loss(q, ina, inb)
                            losses.append(loss_var.item() + loss_boundary.item())
                        mean_score = np.mean(losses)
                        self._validation_scores.append(mean_score)

                        if mean_score < best_valid_score:
                            best_valid_score = mean_score
                            valid_patience_counter = 0
                            best_lobe_state = self._lobe.state_dict()
                        else:
                            valid_patience_counter += 1
                            if valid_patience_counter >= valid_patience:
                                print(f"Validation patience reached at epoch {epoch}")
                                self._lobe.load_state_dict(best_lobe_state)
                                return self

                    if self._save_model_interval is not None:
                        if (epoch + 1) % self._save_model_interval == 0:
                            # save the model with the epoch and the mean score
                            self._save_models.append(
                                (epoch, mean_score, model.state_dict())
                            )

        return self

    def transform(self, dataset, batch_size):
        eps = self._epsilon
        model = self._lobe
        model.eval()
        device = self._device
        model.to(device)

        out_list = []
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for data, _, _ in tqdm(loader):
                data = data.to(device)
                out = model(data)
                out_list.append(out.clone().detach().cpu())

        q = torch.cat(out_list, dim=0)
        q = q.numpy()
        q = np.clip((1 + 2 * eps) * expit(q) - eps, 0, 1)
        return q

    def fetch_model(self):
        from copy import deepcopy

        return deepcopy(self._lobe)

    def save_model(self, path, name="lobe.pt"):
        import os
        torch.save(self._lobe.state_dict(), os.path.join(path, name))
        torch.save(self, os.path.join(path, "vcn.pt"))

        return self._lobe
