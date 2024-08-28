import numpy as np
from typing import Optional, Any, Sequence, Union, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...layers.mlps import MLP


class SPIB(nn.Module):
    """
    Initialize the state autoencoder model for geom2vec-spib

    Args:
        prior_model: The prior model for the latent space.
        intermediate_channels: Number of intermediate channels.
        bottleneck_channels: Number of bottleneck channels.
        output_channels: Number of output channels.
        num_layers: Number of layers in the MLP.
        mlp_out_activation: Activation function for the output layer of the
            MLP. Defaults to None.
        device: Device to use for computation. Defaults to torch.device("cpu").
    """

    def __init__(
            self,
            graph_model: nn.Module,
            update_lables: bool,
            bottleneck_channels: int,
            intermediate_channels: int,
            output_channels: int,
            num_layers: int,
            mlp_out_activation: Optional[nn.Module] = None,
            device: torch.device = torch.device("cpu"),
            optimizer: str = "adam",
            learning_rate: float = 5e-4,
            weight_decay: float = 0.0,
    ):
        super(SPIB, self).__init__()

        self.optimizer_types = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }
        if optimizer == "GrokFastAdamW":
            from grokfast_pytorch import GrokFastAdamW
            self.optimizer_types["GrokFastAdamW"] = GrokFastAdamW
        elif optimizer == "AdamAtan2":
            from adam_atan2_pytorch import AdamAtan2
            self.optimizer_types["AdamAtan2"] = AdamAtan2

        if optimizer not in self.optimizer_types.keys():
            raise ValueError(
                f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}"
            )
        self._optimizer = self.optimizer_types[optimizer](
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        self.update_lables = update_lables

        self.graph_model = graph_model
        self.encoder = MLP(
            input_channels=intermediate_channels,
            hidden_channels=intermediate_channels,
            out_channels=bottleneck_channels,
            num_layers=num_layers,
            out_activation=mlp_out_activation,
        )
        self.decoder = MLP(
            input_channels=bottleneck_channels,
            hidden_channels=intermediate_channels,
            out_channels=intermediate_channels,
            num_layers=num_layers,
            out_activation=mlp_out_activation,
        )
        self.cls_output = nn.Sequential(
            nn.Linear(intermediate_channels, self.output_channels),
            nn.Softmax(dim=-1),
        )
        self.enc_logvar = nn.Parameter(torch.tensor([0.0]))

        self.rep_weights = nn.Sequential(
            nn.Linear(self.output_channels, 1, bias=False),
            nn.Softmax(dim=0)
        )
        self.register_buffer('representation_inputs', torch.eye(self.output_channels, intermediate_channels),
                             device=device, requires_grad=False)  # check this, original code has np.prod(data_shape)

        self.register_buffer('idle_input', torch.eye(self.output_channels, self.output_channels),
                             device=device, requires_grad=False)
        self.bottleneck_channels = bottleneck_channels
        self.to(device)

    def get_prior(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Get the prior distribution of the latent space.
        Args:
            graph_emb: The input graph embedding of shape (batch_size, hidden_channels).

        Returns:
            z_mean: The mean of the latent space of shape (batch_size, z_dim).
            z_logvar: The log variance of the latent space of shape (1).
        """
        # input dim should be (batch, hidden_channels)
        z_mean = self.encoder(graph_emb)
        z_logvar = self.enc_logvar
        return z_mean, z_logvar

    def encode(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encode the input data to the latent space.
        Args:
            data: The input data of shape (batch_size, num_nodes, num_features, feature_dim) for the token mixer.
            drop the -2 dim for the non-mixer case

        Returns:
            z_mean: The mean of the latent space of shape (batch_size, z_dim).
            z_logvar: The log variance of the latent space of shape (1).
        """

        x = self.graph_model(data)
        z_mean, z_logvar = self.get_prior(x)
        return z_mean, z_logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick.
        Args:
            mu: shape: [batch_size, z_dim]
            logvar: initialized as a parameter, shape: [1]

        Returns:
            z: shape: [batch_size, z_dim]

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space to the state label space.
        Args:
            z: The latent space of shape (batch_size, z_dim).

        Returns:
            x: The state label space of shape (batch_size, num_classes).

        """
        # just decode to state labels
        x = self.decoder(z)
        x = self.cls_output(x)
        return x

    def forward(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass of the model.
        Args:
            data:
            shape: [batch_size, num_nodes, num_features, feature_dim] for the token mixer.
            drop the -2 dim for the non-mixer case

        Returns:
            pred_labels: The predicted state labels of shape (batch_size, num_classes).
            z_sample: The sampled latent space of shape (batch_size, z_dim).
            z_mean: The mean of the latent space of shape (batch_size, z_dim).
            z_logvar: The log variance of the latent space of shape (1).

        """
        z_mean, z_logvar = self.encode(data)
        z = self.reparam(z_mean, z_logvar)
        z_sample = self.reparam(z_mean, z_logvar)
        pred_labels = self.decode(z)
        return pred_labels, z_sample, z_mean, z_logvar

    def get_rep_z(self):
        """
        Get the representation of the latent space.
        Returns:
            rep_z_mean: The mean of the latent space of shape (self.output_channels, bottleneck_channels).
            rep_z_logvar: The log variance of the latent space of shape ([1]).
        """
        X = self.representation_inputs  # should be (self.output_channels, intermediate_channels)
        rep_z_mean, rep_z_logvar = self.get_prior(X)
        return rep_z_mean, rep_z_logvar

    def log_p(self, z, sum_up=True):
        """
        Compute the log probability of the latent space.
        Args:
            z: The latent space of shape (batch_size, z_dim).
            sum_up: Whether to sum up the log probability. Defaults to True.

        Returns:

        """
        rep_z_mean, rep_z_logvar = self.get_rep_z()
        w = self.rep_weights(self.idle_input)
        z_expand = z.unsqueeze(1)  # (batch_size, z_dim) -> (batch_size, 1, z_dim)
        rep_mean = rep_z_mean.unsqueeze(
            0)  # (self.output_channels, bottleneck_channels) -> (1, self.output_channels, bottleneck_channels)
        rep_logvar = rep_z_logvar.unsqueeze(0)  # (1) -> (1, 1)
        rep_log_q = -0.5 * torch.sum(rep_logvar + torch.pow(z_expand - rep_mean, 2)
                                     / torch.exp(rep_logvar), dim=2)
        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(rep_log_q) @ w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(rep_log_q) * w.T + self.eps)

        return log_p

    def reset_representative(self, representative_inputs):

        # reset the nuber of representative inputs
        self.output_channels = representative_inputs.shape[0]

        # reset representative weights
        self.idle_input = torch.eye(self.output_channels, self.output_channels, device=self.device, requires_grad=False)

        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_channels, 1, bias=False),
            nn.Softmax(dim=0))
        self.representative_weights[0].weight = nn.Parameter(torch.ones([1, self.output_channels], device=self.device))

        # reset representative inputs
        self.representative_inputs = representative_inputs.clone().detach()

    def compute_loss(self, data, time_lagged_labels, data_weights):

        # pass through VAE
        outputs, z_sample, z_mean, z_logvar = self.forward(data)

        # KL Divergence
        log_p = self.log_p(z_sample)
        log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample - z_mean, 2) / torch.exp(z_logvar), dim=1)

        # Reconstruction loss is cross-entropy
        # reweighed
        reconstruction_error = torch.sum(
            data_weights * torch.sum(-time_lagged_labels * outputs, dim=1)) / data_weights.sum()

        # KL Divergence
        kl_loss = torch.sum(data_weights * (log_q - log_p)) / data_weights.sum()
        loss = reconstruction_error + self.beta * kl_loss

        return loss, reconstruction_error.detach().cpu().data, kl_loss.detach().cpu().data

    def _init_logs(self):
        self.training_loss = []
        self.training_reconst_loss = []
        self.training_kl_loss = []
        self.validation_loss = []
        self.validation_reconst_loss = []
        self.validation_kl_loss = []
        self.state_labels_history = []
        self.state_change_history = []

        self._step = 0

    @torch.no_grad()
    def _infer_new_labels(self, time_lagged_data, batch_size):
        if self.UpdateLabel:
            labels = []

            for i in range(0, len(time_lagged_data), batch_size):
                batch_inputs = time_lagged_data[i:i+batch_size]

                # pass through VAE
                z_mean, z_logvar = self.encode(batch_inputs)
                log_prediction = self.decode(z_mean)

                # label = p/Z
                labels += [log_prediction.exp()]

            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_channels)

            return labels
    @torch.no_grad()
    def update_model(self,data,data_weights):
        pass

    def _log_model_params(self):
        self.model_params = {
            "graph_model": self.graph_model.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "cls_output": self.cls_output.state_dict(),
            "enc_logvar": self.enc_logvar,
            "rep_weights": self.rep_weights.state_dict(),
            "representation_inputs": self.representation_inputs,
            "idle_input": self.idle_input,
            "bottleneck_channels": self.bottleneck_channels,
            "output_channels": self.output_channels,
        }

    def _restore_model_params(self):
        # restore the parameters of the model
        self.graph_model.load_state_dict(self.model_params["graph_model"])
        self.encoder.load_state_dict(self.model_params["encoder"])
        self.decoder.load_state_dict(self.model_params["decoder"])
        self.cls_output.load_state_dict(self.model_params["cls_output"])
        self.enc_logvar = self.model_params["enc_logvar"]
        self.rep_weights.load_state_dict(self.model_params["rep_weights"])

    def fit(
            self,
            train_dataset,
            val_dataset,
            batch_size: int = 5000,
            max_updates: int = 15,
            tolerance: float = 1e-2,
            n_epochs: int = 1,
            progress: Any = tqdm,
            mask_threshold: float = 0.0,
            train_patience: int = 1000,
            valid_patience: int = 1000,
            train_valid_interval: int = 1000,
    ):
        """Performs fit on data.

        Parameters
        ----------
        train_loader
            Yield a tuple of batches representing instantaneous and time-lagged state labels for training.
        n_epochs
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader
             Yield a tuple of batches representing instantaneous and time-lagged state labels for validation.
        progress

        Returns
        -------
        self
        """

        self._init_logs()
        self._log_model_params()
        self._step = 0

        update_counter = 0
        best_train_loss = np.inf
        best_valid_loss = np.inf
        train_patience_counter = 0
        valid_patience_counter = 0

        init_state_pop = (torch.sum(train_dataset.time_lagged_labels, dim=0).float() / train_dataset.time_lagged_labels.shape[0]).cpu()
        self.state_labels_history.append(init_state_pop.numpy())

        train_dataset = SPIBDataset(train_dataset[0], train_dataset[1], train_dataset[2], train_dataset[3])
        val_dataset = SPIBDataset(val_dataset[0], val_dataset[1], val_dataset[2], val_dataset[3])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_dataset is not None:
            validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        for epoch in progress(
                range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            for batch_0, weight_0, lable_1 in tqdm(train_loader):
                self._step += 1
                loss, reconstruction_error, kl_loss = self.calculate_loss(batch_0, lable_1, weight_0)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                weight_sum = weight_0.sum().cpu()
                self.training_loss.append(loss.item() / weight_sum)
                self.training_reconst_loss.append(reconstruction_error.item() / weight_sum)
                self.training_kl_loss.append(kl_loss.item() / weight_sum)

                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    train_patience_counter = 0
                else:
                    train_patience_counter += 1
                    if train_patience_counter > train_patience:
                        print("Early stopping due to no improvement in training loss.")
                        self._restore_model_params()
                        return self

                if (
                        validation_loader is not None
                        and self._step % train_valid_interval == 0
                ):
                    for batch_0, weight_0, lable_1 in tqdm(validation_loader):
                        loss, reconstruction_error, kl_loss = self.calculate_loss(batch_0, lable_1, weight_0)
                        weight_sum = weight_0.sum().cpu()
                        self.validation_loss.append(loss.item() / weight_sum)
                        self.validation_reconst_loss.append(reconstruction_error.item() / weight_sum)
                        self.validation_kl_loss.append(kl_loss.item() / weight_sum)

                        if loss.item() < best_valid_loss:
                            best_valid_loss = loss.item()
                            valid_patience_counter = 0
                            self._log_model_params()  # save the best model
                        else:
                            valid_patience_counter += 1
                            if valid_patience_counter > valid_patience:
                                print("Early stopping due to no improvement in validation loss.")
                                self._restore_model_params()
                                return self

                        new_train_labels = self._infer_new_labels(train_dataset.time_lagged_data, batch_size)
                        train_dataset.update_labels(new_train_labels)
                        state_population = (torch.sum(new_train_labels, dim=0).float()/new_train_labels.shape[0]).cpu()
                        self.state_labels_history.append(state_population.numpy())
                        print(f"State population: {state_population.numpy()}")
                        # print the relative state population change
                        mask = (init_state_pop > mask_threshold)
                        del_state_pop = torch.sqrt(
                            torch.square((state_population - init_state_pop)[mask] / init_state_pop[mask]).mean())
                        print(f"Relative state population change: {del_state_pop.item()}")
                        self.state_change_history.append(del_state_pop.item())
                        init_state_pop = state_population

                        if torch.sum(state_population > 0) < 2:
                            print("Only one metastable state is found!")
                            raise ValueError

                        if self.update_lables and update_counter < max_updates:
                            update_counter += 1

                            train_labels = new_train_labels
                            val_labels = self._infer_new_labels(val_dataset.time_lagged_data, batch_size)
                            train_labels.to(self.device)
                            val_labels.to(self.device)

                            print('Updated labels')











from torch.utils.data import Dataset


class SPIBDataset(Dataset):
    def __init__(self, data, data_weights, future_data, time_lagged_labels):
        self.data = data
        self.data_weights = data_weights
        self.time_lagged_labels = time_lagged_labels
        self.time_lagged_data = future_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_weights[idx], self.time_lagged_labels[idx]

    def update_labels(self, new_labels):
        if not isinstance(new_labels, torch.Tensor):
            new_labels = torch.tensor(new_labels)

        # Update the labels directly in-place
        self.time_lagged_labels[:] = new_labels
