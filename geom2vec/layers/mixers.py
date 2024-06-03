import torch
import torch.nn as nn
import copy
from einops.layers.torch import Rearrange, Reduce
from torch.nn import TransformerEncoderLayer, TransformerEncoder


### Transformer on coarsed graph, i.e. residue level (SubFormer)
class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attention_weights = []

        for layer in self.layers:
            output, attn_weights = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attention_weights.append(attn_weights)

        return output, attention_weights


class SubFormer(nn.Module):
    r"""
    Transformer model on coarsed graph, i.e. residue level.

    Args:
        hidden_channels (int): The number of hidden units.
        encoder_layers (int): The number of encoder layers.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        pool (str): The pooling method.
        dropout (float): The dropout rate.
        attn_map (bool): Whether to return the attention map. i.e. using the custom transformer encoder layer.
    """

    def __init__(
            self,
            hidden_channels,
            encoder_layers,
            nhead,
            dim_feedforward,
            pool="cls",
            dropout=0.1,
            attn_map=True,
            attn_mask=None,
            pool_mask=None,
    ):
        super(SubFormer, self).__init__()

        ## transformer encoder
        self.attn_map = attn_map
        if attn_map:
            encoder_layer = CustomTransformerEncoderLayer(
                batch_first=True,
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.transformer_encoder = CustomTransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )
        else:
            encoder_layer = TransformerEncoderLayer(
                batch_first=True,
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        if attn_mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
            # suppose user input is (seq_len) true/false mask or additive mask
            if attn_mask.dtype == torch.bool:
                # create the square attention mask, True means masked and not allowed to attend
                # add a False to the first element of the mask, so that the cls token can attend to all nodes
                attn_mask = torch.cat([torch.tensor([False]), attn_mask], dim=0)
                # seq_len -> (seq_len, seq_len)
                attn_mask = attn_mask.unsqueeze(0) | attn_mask.unsqueeze(1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float(0.0)).masked_fill(attn_mask == 1,
                                                                                                  float('-inf'))

            elif torch.is_floating_point(attn_mask):
                # additive weight to the attention score, can bias the attention score

                # add a 0 to the first element of the mask, so that the cls token is not affected by the mask
                attn_mask = torch.cat([torch.tensor([0.0]), attn_mask], dim=0)
                attn_mask = attn_mask.unsqueeze(0) + attn_mask.unsqueeze(1)

            else:
                raise ValueError("attn_mask must be a boolean tensor or a float tensor")

        self.attn_mask = attn_mask

        if pool_mask is not None:
            # add a False to the first element of the mask for the cls token
            self.pool_mask = torch.cat([torch.tensor([False]), pool_mask], dim=0)
            assert self.pool != "cls", "pool_mask is not compatible with cls token pooling"

    def get_weights(self, data):
        src = data
        src = torch.cat((self.cls_token.expand(src.size(0), -1, -1), src), dim=1)
        src, attention_weights = self.transformer_encoder(
            src,
            mask=self.attn_mask,
            src_key_padding_mask=None
        )
        return attention_weights

    def forward(self, data):
        src = data
        src = torch.cat((self.cls_token.expand(src.size(0), -1, -1), src), dim=1)

        if self.attn_map:
            src, _ = self.transformer_encoder(
                src,
                mask=self.attn_mask,
                src_key_padding_mask=None
            )
        else:
            src = self.transformer_encoder(src,
                                           mask=self.attn_mask,
                                           src_key_padding_mask=None)

        if hasattr(self, "pool_mask"):
            src = src[:, ~self.pool_mask, :]  # remove the masked patches

            if self.pool == "mean":
                out = src[:, 1:, :].mean(dim=1)
            elif self.pool == "sum":
                out = src[:, 1:, :].sum(dim=1)

        else:
            if self.pool == "cls":
                out = src[:, 0, :]
            elif self.pool == "mean":
                out = src[:, 1:, :].mean(dim=1)
            elif self.pool == "sum":
                out = src[:, 1:, :].sum(dim=1)

        return out


# MLP-Mixer on coarsed graph, i.e. residue level (SubMixer)


class FeedForward(nn.Module):
    r"""Feedforward neural network with GELU activation.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        dropout (float): The dropout rate.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    r"""Mixer block.

    Args:
        dim (int): The number of input features.
        num_patch (int): The number of patches.
        token_dim (int): The number of token dimensions.
        channel_dim (int): The number of channel dimensions.
        dropout (float): The dropout rate.
    """

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n d -> b d n"),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange("b d n -> b n d"),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class SubMixer(nn.Module):
    r"""Mixer model.

    Args:
        num_patch (int): The number of patches.
        depth (int): The number of mixer blocks.
        dropout (float): The dropout rate.
        dim (int): The number of input features.
        token_dim (int): The number of token dimensions.
        channel_dim (int): The number of channel dimensions.
        pooling (str): The pooling method.
    """

    def __init__(
            self,
            num_patch,
            depth,
            dropout,
            dim,
            token_dim,
            channel_dim,
            pool="mean",
            pool_mask=None,
    ):
        super().__init__()
        self.num_patch = num_patch
        self.depth = depth
        self.mixer = nn.Sequential(
            *[
                MixerBlock(dim, num_patch, token_dim, channel_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.pool = pool
        if pool not in ["mean", "sum"]:
            raise ValueError(
                f"Pooling should be either 'mean' or 'sum' but got {pool}"
            )

        if pool_mask is not None:
            assert (
                    pool_mask.shape[0] == num_patch
            ), f"Input tensor has {pool_mask.shape[0]} patches, but expected {num_patch}"

            # assume pool_mask is a boolean tensor
            assert (
                    pool_mask.dtype == torch.bool
            ), "pool_mask must be a boolean tensor"

            self.pool_mask = pool_mask

    def forward(self, x):
        assert (
                x.shape[1] == self.num_patch
        ), f"Input tensor has {x.shape[1]} patches, but expected {self.num_patch}"
        x = self.mixer(x)

        if hasattr(self, "pool_mask"):
            x = x[:, ~self.pool_mask, :]  # remove the masked patches

            if self.pool == "mean":
                x = x.mean(1)
            elif self.pool == "sum":
                x = x.sum(1)

        else:
            if self.pool == "mean":
                x = x.mean(1)
            elif self.pool == "sum":
                x = x.sum(1)

        return x
