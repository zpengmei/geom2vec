import torch
import numpy as np
from tqdm import tqdm


class VarComm(torch.nn.Module):
    r"""
    Variational committor network for estimating the committor function.
    """

    def __init__(self,
                 lobe,
                 optimizer='adam',
                 device='cuda',
                 learning_rate=5e-4,
                 epsilon=1e-1,
                 k=10,
                 save_model_interval=None,
                 ):
        super(VarComm, self).__init__()

        self._lobe = lobe
        self._learning_rate = learning_rate
        self._device = torch.device(device)
        self._epsilon = epsilon
        self._k = k

        assert optimizer in ['adam', 'adamw', 'sgd']

        if optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'adamw':
            self._optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

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
            The network output.
        ina: torch.Tensor
            The input data for the first state.
        inb: torch.Tensor
            The input data for the second state.

        Returns
        -------

        """

        k = self._k
        eps = self._epsilon

        x = x * (1 + eps)
        q = torch.clip(x, min=0, max=1)
        loss_var = torch.mean(0.5 * torch.diff(q, dim=0) ** 2)
        loss_boundary = torch.mean(k * (x[ina] + eps) ** 2)
        loss_boundary += torch.mean(k * (x[inb] - (1 + eps)) ** 2)
        return loss_var, loss_boundary

    def fit(self, train_loader, n_epochs=1, val_loader=None, progress=tqdm):
        r"""
        Fit the committor network to the training data.

        Parameters
        ----------
        train_loader
        n_epochs
        val_loader
        progress

        Returns
        -------

        """
        self._step = 0

        for epoch in progress(range(n_epochs), desc='epoch', total=n_epochs, leave=False):

            optimizer = self._optimizer
            model = self._lobe
            k = self._k

            model.train()

            for data, ina, inb in progress(train_loader, desc='batch', total=len(train_loader), leave=False):
                optimizer.zero_grad()
                data = data.to(self._device)
                ina = ina.to(self._device)
                inb = inb.to(self._device)

                q = model(data)
                loss_var, loss_boundary = self._variational_loss(q, ina, inb)
                loss = loss_var + loss_boundary
                loss.backward()
                optimizer.step()

                self._training_scores.append(loss_var.item() + loss_boundary.item() / k)

            if val_loader is not None:
                with torch.no_grad():
                    model.eval()

                    losses = []
                    for data, ina, inb in progress(val_loader, desc='validation', total=len(val_loader), leave=False):
                        data = data.to(self._device)
                        ina = ina.to(self._device)
                        inb = inb.to(self._device)
                        q = model(data)
                        loss_var, loss_boundary = self._variational_loss(q, ina, inb)
                        losses.append(loss_var.item() + loss_boundary.item() / k)
                    mean_score = np.mean(losses)
                    self._validation_scores.append(mean_score)

                if self._save_model_interval is not None:
                    if (epoch + 1) % self._save_model_interval == 0:
                        # save the model with the epoch and the mean score
                        self._save_models.append((epoch, mean_score, model.state_dict()))

        return self

    def transform(self, dataset, batch_size):

        model = self._lobe
        model.eval()
        device = self._device
        model.to(device)

        out_list = []
        with torch.no_grad():
            batch_dataset = torch.split(dataset, batch_size)
            for data in tqdm(batch_dataset):
                data = data.to(device)
                out = model(data)
                out_list.append(out.clone().detach().cpu())

        comm = torch.cat(out_list, dim=0)
        comm = comm.numpy()
        comm = np.clip(1.1 * comm, 0, 1)
        return comm
