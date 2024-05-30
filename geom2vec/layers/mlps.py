from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Linear


class GatedEquivariantBlock(torch.nn.Module):
    r"""Applies a gated equivariant operation to scalar features and vector
    features from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int, optional): The number of channels in the
            intermediate layer, or :obj:`None` to use the same number as
            :obj:`hidden_channels`. (default: :obj:`None`)
        scalar_activation (bool, optional): Whether to apply a scalar
            activation function to the output node features.
            (default: obj:`False`)
    """

    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            intermediate_channels: Optional[int] = None,
            scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = Linear(hidden_channels, out_channels, bias=False)

        self.update_net = torch.nn.Sequential(
            Linear(hidden_channels * 2, intermediate_channels),
            torch.nn.SiLU(),
            Linear(intermediate_channels, out_channels * 2),
        )

        self.act = torch.nn.SiLU() if scalar_activation else None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.zero_()

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Applies a gated equivariant operation to node features and vector
        features.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.
        """
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantScalar(torch.nn.Module):
    r"""Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()

        self.output_network = torch.nn.ModuleList([
            GatedEquivariantBlock(
                hidden_channels,
                hidden_channels // 2,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                hidden_channels // 2,
                out_channels,
                scalar_activation=False,
            ),
        ])

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tensor:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0, v


class MLP(torch.nn.Module):
    r"""A multi-layer perceptron (MLP) with SiLU activation functions."""

    def __init__(self,
                 input_channels:int,
                 hidden_channels:int,
                 out_channels: int,
                 num_layers: int,
                 out_activation: Optional[torch.nn.Module] = None,
                 ) -> None:
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_channels, hidden_channels))
        self.layers.append(torch.nn.SiLU())
        for _ in range(num_layers-2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(torch.nn.SiLU())
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))

        if out_activation is not None:
            self.out_activation = out_activation
        else:
            self.out_activation = torch.nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        r"""Applies the MLP to the input tensor."""
        for layer in self.layers:
            x = layer(x)
        return self.out_activation(x)