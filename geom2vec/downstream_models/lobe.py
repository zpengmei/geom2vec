import torch
import torch.nn as nn
from geom2vec.layers.mlps import EquivariantScalar, MLP
from geom2vec.layers.mixers import SubFormer, SubMixer
from typing import Optional
from torch import Tensor


class Lobe(torch.nn.Module):
    def __init__(
            self,
            # general parameters
            hidden_channels: int,
            intermediate_channels: int,
            output_channels: int,
            num_layers: int,
            batch_norm: bool = False,
            vector_feature: bool = True,
            mlp_dropout: float = 0.0,
            mlp_out_activation=Optional[nn.Module],
            device: torch.device = torch.device("cpu"),
            # mixer parameters
            token_mixer: str = "none",  # None, subformer, submixer
            num_mixer_layers: int = 4,  # number of layers for transformer or mlp-mixer
            expansion_factor: int = 2,  # expansion factor for transformer FF
            nhead: int = 8,  # number of heads for transformer
            pooling: str = "cls",  # cls, mean, sum
            dropout: float = 0.1,  # dropout for mlp-mixer and transformer
            attn_map: bool = False,  # whether to return attention map of transformer
            # suppose user input is (seq_len) true/false mask, True means masked and not to be pooled
            num_tokens: int = 1,  # number of tokens for mlp-mixer
            token_dim: int = 64,  # dimension of tokens for mlpixer
            #### make sure you know what you are doing when using masks ####
            attn_mask: Tensor = None,  # attention mask for transformer (optional)
            pool_mask: Tensor = None,  # pool mask for transformer/MLP-mixer (optional)
            #################################################################
    ):
        super(Lobe, self).__init__()

        assert token_mixer in ["none", "subformer", "submixer"]
        assert pooling in ["cls", "mean", "sum"]

        if token_mixer == "submixer" and pooling == "cls":
            raise ValueError("Submixer does not support cls pooling")

        self.pooling = pooling

        self.token_mixer = token_mixer
        self.vector_feature = vector_feature
        self.dropout = torch.nn.Dropout(mlp_dropout)

        self.input_projection = EquivariantScalar(
            hidden_channels, intermediate_channels
        )
        if not vector_feature:
            self.input_projection = torch.nn.Linear(
                hidden_channels, intermediate_channels
            )

        attn_mask = attn_mask.to(device) if attn_mask is not None else None
        pool_mask = pool_mask.to(device) if pool_mask is not None else None

        if token_mixer == "none":
            self.mixer = None

        elif token_mixer == "subformer":
            self.mixer = SubFormer(
                hidden_channels=intermediate_channels,
                encoder_layers=num_mixer_layers,
                nhead=nhead,
                dim_feedforward=int(expansion_factor * intermediate_channels),
                pool=pooling,
                dropout=dropout,
                attn_map=attn_map,
                attn_mask=attn_mask,
                pool_mask=pool_mask,
                device=device,
            )
        elif token_mixer == "submixer":
            self.mixer = SubMixer(
                num_patch=num_tokens,
                depth=num_mixer_layers,
                dropout=dropout,
                dim=intermediate_channels,
                token_dim=token_dim,
                channel_dim=int(expansion_factor * intermediate_channels),
                pool=pooling,
                pool_mask=pool_mask,
                device=device,
            )

        self.output_projection = MLP(
            input_channels=intermediate_channels,
            hidden_channels=intermediate_channels,
            out_channels=output_channels,
            num_layers=num_layers,
            out_activation=mlp_out_activation,
        )

        self.batch_norm = batch_norm

        if batch_norm:
            self.batchnorm = nn.BatchNorm1d(intermediate_channels)

        self.to(device)

    def forward(self, data):
        # several assumptions:
        # 1. if no mixer is used, input shape is (batch,4,hidden_channels)
        # 2. if mixer is used, input shape is (batch,token,4,hidden_channels)
        # 3. the first dim of 4 is scalar feature, the rest are vector features

        assert data.shape[-2] == 4

        x = None
        if self.token_mixer == "none":
            x_rep = data[:, 0, :]
            v_rep = data[:, 1:, :]

            if not self.vector_feature:
                x_rep = self.input_projection(x_rep)
            else:
                x_rep, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)

            x_rep = self.dropout(x_rep)
            if self.batch_norm:
                x_rep = self.batchnorm(x_rep)
            x = self.output_projection(x_rep)

        elif self.token_mixer == "subformer":
            batch_size, num_nodes, _, _ = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            if not self.vector_feature:
                x = self.input_projection(x_rep)
            else:
                x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, -1)
            x = self.mixer(x)
            x = self.output_projection(x)

        elif self.token_mixer == "submixer":
            batch_size, num_nodes, _, dim = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            if not self.vector_feature:
                x = self.input_projection(x_rep)
            else:
                x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, dim)
            x = self.mixer(x)
            x = self.output_projection(x)

        return x

    def fetch_attnmap(self, data):
        assert self.token_mixer == "subformer"
        batch_size, num_nodes, _, _ = data.shape
        x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
        v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
        if not self.vector_feature:
            x = self.input_projection(x_rep)
        else:
            x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
        x = x.reshape(batch_size, num_nodes, -1)
        attn_map = self.mixer.get_weights(x)
        return attn_map

