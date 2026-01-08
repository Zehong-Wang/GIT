from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.utils import spmm


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def __init__(self, hidden_dim=None, output_dim=None):
        super().__init__()
        if hidden_dim is not None:
            self.proj_z = True
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        z = self.lin(z) if self.proj_z else z
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)

            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


# From pyg implementation
class MySAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False,
                 root_weight=True, project=False, bias=True, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_attr=None):
        # Does not process edge attributes
        if edge_attr is not None:
            return (x_j + edge_attr).relu()
        else:
            return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


class NonParamPooling(MessagePassing):
    def __init__(self, aggr="mean"):
        super().__init__(aggr)

    def forward(self, x, edge_index, edge_attr=None):

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        # Does not process edge attributes
        return x_j
        # return (x_j + xe).relu()

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation, num_layers, backbone='sage',
                 normalize='none', dropout=0.0, act_first=True):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize
        self.act_first = act_first

        self.activation = activation()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        dims = [input_dim] + [hidden_dim] * num_layers

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            if backbone == 'sage':
                self.layers.append(MySAGEConv(in_dim, out_dim, aggr='mean', normalize=False, root_weight=True))
            elif backbone == 'gat':
                self.layers.append(GATConv(in_dim, out_dim // 4, heads=4))
            self.norms.append(nn.BatchNorm1d(out_dim))

        self.mean_aggr = NonParamPooling(aggr='mean')
        self.pooling_lin = nn.Linear(hidden_dim, hidden_dim)
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.pooling_lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return z

    def encode(self, x, edge_index, edge_attr=None):
        z = x

        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_attr)
            if self.act_first:
                z = self.activation(z)
            if self.normalize != 'none':
                z = self.norms[i](z)
            if not self.act_first:
                z = self.activation(z)

            z = self.dropout(z)

        return z

    def encode_graph(self, x, edge_index, batch=None, pool="mean"):
        z = self.encode(x, edge_index)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def pooling(self, x, edge_index):
        z = self.mean_aggr(x, edge_index)
        z = self.pooling_lin(z)
        # z = self.norms[-1](z)

        return z

    def save(self, path):
        torch.save(self.state_dict(), path)
