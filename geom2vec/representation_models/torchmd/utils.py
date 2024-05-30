import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph



def visualize_basis(basis_type, num_rbf=50, cutoff_lower=0, cutoff_upper=5):
    """
    Function for quickly visualizing a specific basis. This is useful for inspecting
    the distance coverage of basis functions for non-default lower and upper cutoffs.

    Args:
        basis_type (str): Specifies the type of basis functions used. Can be one of
            ['gauss',expnorm']
        num_rbf (int, optional): The number of basis functions.
            (default: :obj:`50`)
        cutoff_lower (float, optional): The lower cutoff of the basis.
            (default: :obj:`0`)
        cutoff_upper (float, optional): The upper cutoff of the basis.
            (default: :obj:`5`)
    """
    import matplotlib.pyplot as plt

    distances = torch.linspace(cutoff_lower - 1, cutoff_upper + 1, 1000)
    basis_kwargs = {
        "num_rbf": num_rbf,
        "cutoff_lower": cutoff_lower,
        "cutoff_upper": cutoff_upper,
    }
    basis_expansion = rbf_class_mapping[basis_type](**basis_kwargs)
    expanded_distances = basis_expansion(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


# from ssp.modules.models.representation_models.torchmd.extensions import get_neighbor_pairs_kernel
# class OptimizedDistance(torch.nn.Module):
#     """ Compute the neighbor list for a given cutoff.
#
#         This operation can be placed inside a CUDA graph in some cases.
#         In particular, resize_to_fit and check_errors must be False.
#
#         Note that this module returns neighbors such that :math:`r_{ij} \\ge \\text{cutoff_lower}\\quad\\text{and}\\quad r_{ij} < \\text{cutoff_upper}`.
#
#         This function optionally supports periodic boundary conditions with
#         arbitrary triclinic boxes.  The box vectors `a`, `b`, and `c` must satisfy
#         certain requirements:
#
#         .. code:: python
#
#            a[1] = a[2] = b[2] = 0
#            a[0] >= 2*cutoff, b[1] >= 2*cutoff, c[2] >= 2*cutoff
#            a[0] >= 2*b[0]
#            a[0] >= 2*c[0]
#            b[1] >= 2*c[1]
#
#         These requirements correspond to a particular rotation of the system and
#         reduced form of the vectors, as well as the requirement that the cutoff be
#         no larger than half the box width.
#
#         Parameters
#         ----------
#         cutoff_lower : float
#             Lower cutoff for the neighbor list.
#         cutoff_upper : float
#             Upper cutoff for the neighbor list.
#         max_num_pairs : int
#             Maximum number of pairs to store, if the number of pairs found is less than this, the list is padded with (-1,-1) pairs up to max_num_pairs unless resize_to_fit is True, in which case the list is resized to the actual number of pairs found.
#             If the number of pairs found is larger than this, the pairs are randomly sampled. When check_errors is True, an exception is raised in this case.
#             If negative, it is interpreted as (minus) the maximum number of neighbors per atom.
#         strategy : str
#             Strategy to use for computing the neighbor list. Can be one of :code:`["shared", "brute", "cell"]`.
#
#             1. *Shared*: An O(N^2) algorithm that leverages CUDA shared memory, best for large number of particles.
#             2. *Brute*: A brute force O(N^2) algorithm, best for small number of particles.
#             3. *Cell*:  A cell list algorithm, best for large number of particles, low cutoffs and low batch size.
#         box : torch.Tensor, optional
#             The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)` if a ox per sample is desired.
#             where `box_vectors[0] = a`, `box_vectors[1] = b`, and `box_vectors[2] = c`.
#             If this is omitted, periodic boundary conditions are not applied.
#         loop : bool, optional
#             Whether to include self-interactions.
#             Default: False
#         include_transpose : bool, optional
#             Whether to include the transpose of the neighbor list.
#             Default: True
#         resize_to_fit : bool, optional
#             Whether to resize the neighbor list to the actual number of pairs found. When False, the list is padded with (-1,-1) pairs up to max_num_pairs
#             Default: True
#             If this is True the operation is not CUDA graph compatible.
#         check_errors : bool, optional
#             Whether to check for too many pairs. If this is True the operation is not CUDA graph compatible.
#             Default: True
#         return_vecs : bool, optional
#             Whether to return the distance vectors.
#             Default: False
#         long_edge_index : bool, optional
#             Whether to return edge_index as int64, otherwise int32.
#             Default: True
#         """
#     def __init__(
#         self,
#         cutoff_lower=0.0,
#         cutoff_upper=5.0,
#         max_num_pairs=-32,
#         return_vecs=False,
#         loop=False,
#         strategy="brute",
#         include_transpose=True,
#         resize_to_fit=True,
#         check_errors=True,
#         box=None,
#         long_edge_index=True
#     ):
#         super(OptimizedDistance, self).__init__()
#         self.cutoff_upper = cutoff_upper
#         self.cutoff_lower = cutoff_lower
#         self.max_num_pairs = max_num_pairs
#         self.strategy = strategy
#         self.box: Optional[Tensor] = box
#         self.loop = loop
#         self.return_vecs = return_vecs
#         self.include_transpose = include_transpose
#         self.resize_to_fit = resize_to_fit
#         self.use_periodic = True
#         if self.box is None:
#             self.use_periodic = False
#             self.box = torch.empty((0, 0))
#             if self.strategy == "cell":
#                 # Default the box to 3 times the cutoff, really inefficient for the cell list
#                 lbox = cutoff_upper * 3.0
#                 self.box = torch.tensor([[lbox, 0, 0], [0, lbox, 0], [0, 0, lbox]], device="cpu")
#         if self.strategy == "cell":
#             self.box = self.box.cpu()
#         self.check_errors = check_errors
#         self.long_edge_index = long_edge_index
#
#     def forward(
#             self, pos: Tensor, batch: Optional[Tensor] = None, box: Optional[Tensor] = None
#     ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
#         """
#         Compute the neighbor list for a given cutoff.
#
#         Parameters
#         ----------
#         pos : torch.Tensor
#             A tensor with shape (N, 3) representing the positions.
#         batch : torch.Tensor, optional
#             A tensor with shape (N,). Defaults to None.
#         box : torch.Tensor, optional
#             The vectors defining the periodic box.  This must have shape `(3, 3)` or `(max(batch)+1, 3, 3)`,
#         Returns
#         -------
#         edge_index : torch.Tensor
#             List of neighbors for each atom in the batch.
#             Shape is (2, num_found_pairs) or (2, max_num_pairs).
#         edge_weight : torch.Tensor
#             List of distances for each atom in the batch.
#             Shape is (num_found_pairs,) or (max_num_pairs,).
#         edge_vec : torch.Tensor, optional
#             List of distance vectors for each atom in the batch.
#             Shape is (num_found_pairs, 3) or (max_num_pairs, 3).
#
#         Notes
#         -----
#         If `resize_to_fit` is True, the tensors will be trimmed to the actual number of pairs found.
#         Otherwise, the tensors will have size `max_num_pairs`, with neighbor pairs (-1, -1) at the end.
#         """
#         use_periodic = self.use_periodic
#         if not use_periodic:
#             use_periodic = box is not None
#         box = self.box if box is None else box
#         assert box is not None, "Box must be provided"
#         box = box.to(pos.dtype)
#         max_pairs : int = self.max_num_pairs
#         if self.max_num_pairs < 0:
#             max_pairs = -self.max_num_pairs * pos.shape[0]
#         if batch is None:
#             batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
#         edge_index, edge_vec, edge_weight, num_pairs = get_neighbor_pairs_kernel(
#             strategy=self.strategy,
#             positions=pos,
#             batch=batch,
#             max_num_pairs=int(max_pairs),
#             cutoff_lower=self.cutoff_lower,
#             cutoff_upper=self.cutoff_upper,
#             loop=self.loop,
#             include_transpose=self.include_transpose,
#             box_vectors=box,
#             use_periodic=use_periodic,
#         )
#         if self.check_errors:
#             if num_pairs[0] > max_pairs:
#                 raise AssertionError(
#                     "Found num_pairs({}) > max_num_pairs({})".format(
#                         num_pairs[0], max_pairs
#                     )
#                 )
#         # Remove (-1,-1)  pairs
#         if self.resize_to_fit:
#             mask = edge_index[0] != -1
#             edge_index = edge_index[:, mask]
#             edge_weight = edge_weight[mask]
#             edge_vec = edge_vec[mask, :]
#         if self.long_edge_index:
#             edge_index = edge_index.to(torch.long)
#         if self.return_vecs:
#             return edge_index, edge_weight, edge_vec
#         else:
#             return edge_index, edge_weight, None
