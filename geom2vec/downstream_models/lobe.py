import torch
import torch.nn as nn
from geom2vec.layers.mlps import EquivariantScalar, MLP
from geom2vec.layers.mixers import SubFormer, SubMixer


class lobe(torch.nn.Module):

    def __init__(self,
                 # general parameters
                 hidden_channels:int,
                 intermediate_channels:int,
                 output_channels:int,
                 num_layers,
                 batch_norm=False,
                 vector_feature=True,
                 mlp_dropout=0.0,
                 # mixer parameters
                 token_mixer='None',
                 num_mixer_layers=4,
                 expansion_factor=2,
                 nhead=8,
                 pooling='cls',
                 dropout=0.1,
                 attn_map=False,
                 num_tokens=1, # number of tokens for mixer
                 token_dim=64, # dimension of tokens for mixer
                 ):
        super(lobe, self).__init__()

        assert token_mixer in ['None', 'subformer', 'submixer']
        assert pooling in ['cls', 'mean', 'sum']

        if token_mixer == 'submixer' and pooling == 'cls':
            raise ValueError('Submixer does not support cls pooling')

        self.token_mixer = token_mixer
        self.vector_feature = vector_feature
        self.dropout = torch.nn.Dropout(mlp_dropout)

        self.input_projection = EquivariantScalar(hidden_channels, intermediate_channels)
        if not vector_feature:
            self.output_projection = torch.nn.Linear(hidden_channels, intermediate_channels)

        if token_mixer == 'None':
            self.mixer = None
        elif token_mixer == 'subformer':
            self.mixer = SubFormer(
                hidden_channels=intermediate_channels,
                encoder_layers=num_mixer_layers,
                nhead=nhead,
                dim_feedforward=int(expansion_factor*intermediate_channels),
                pool=pooling,
                dropout=dropout,
                attn_map=attn_map
            )
        elif token_mixer == 'submixer':
            self.mixer = SubMixer(
                num_patch=num_tokens,
                depth=num_mixer_layers,
                dropout=dropout,
                dim=intermediate_channels,
                token_dim=token_dim,
                channel_dim=intermediate_channels,
                pooling=pooling
            )

        self.output_projection = MLP(
            input_channels=intermediate_channels,
            hidden_channels=intermediate_channels//2,
            out_channels=output_channels,
            num_layers=num_layers
        )

        self.batch_norm = batch_norm
        if batch_norm:
            self.batchnorm = nn.BatchNorm1d(intermediate_channels)

    def forward(self, data):

        # several assumptions:
        # 1. if no mixer is used, input shape is (batch,4,hidden_channels)
        # 2. if mixer is used, input shape is (batch,token,4,hidden_channels)
        # 3. the first dim of 4 is scalar feature, the rest are vector features

        assert data.shape[-2] == 4

        if self.token_mixer == 'None':
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

        elif self.token_mixer == 'subformer':
            batch_size, num_nodes, _, _ = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, -1)
            x = self.mixer(x)
            x = self.output_projection(x)

        elif self.token_mixer == 'submixer':
            batch_size, num_nodes, _, dim = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, dim)
            x = self.mixer(x)
            if self.pooling == 'mean':
                x = x.mean(1)
            elif self.pooling == 'sum':
                x = x.sum(1)
            x = self.output_projection(x)

        return x







