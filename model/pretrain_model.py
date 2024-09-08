from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling, mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch

from model.encoder import MLP
from utils.utils import seed_everything, get_scheduler, get_device_from_model, check_path

EPS = 1e-15


class PretrainModel(nn.Module):
    def __init__(self, encoder, feat_decoder, topo_decoder):
        super().__init__()

        self.encoder = encoder

        self.feat_decoder = feat_decoder
        self.topo_decoder = topo_decoder

        self.sem_encoder = deepcopy(self.encoder)
        # self.sem_decoder = nn.Linear(self.encoder.hidden_dim, self.encoder.hidden_dim)
        self.sem_decoder = MLP(self.encoder.hidden_dim, self.encoder.hidden_dim, self.encoder.hidden_dim, 2, 0.5)

    def save_encoder(self, path):
        self.encoder.save(path)

    def feat_recon_loss(self, z, x, bs=None):
        z = self.feat_decoder(z)
        return F.mse_loss(z[:bs], x[:bs])

    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):
        if ratio == 0.0:
            # Does not do topological reconstruction
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            # Randomly sample positive edges
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1))
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_loss = -torch.log(self.topo_decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.topo_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def sem_recon_loss(self, z1, z2, eta=1.0, bs=None):
        h1 = self.sem_decoder(z1)
        h2 = self.sem_decoder(z2)

        z1 = F.normalize(z1[:bs], dim=-1, p=2).detach()  # N * D
        z2 = F.normalize(z2[:bs], dim=-1, p=2).detach()
        h1 = F.normalize(h1[:bs], dim=-1, p=2)  # N * D
        h2 = F.normalize(h2[:bs], dim=-1, p=2)

        loss = ((1 - (z1 * h2).sum(dim=-1)).pow_(eta) + (1 - (z2 * h1).sum(dim=-1)).pow_(eta)) / 2
        loss = loss.mean()

        return loss

    def ema_update_sem_encoder(self, decay=0.99):
        for param_q, param_k in zip(self.encoder.parameters(), self.sem_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1 - decay)

    def forward(self, graph, aug_g1, aug_g2, bs=None, **kwargs):
        params = kwargs['params']
        encoder = self.encoder
        device = get_device_from_model(self)

        x, edge_index = graph[0], graph[1]
        x1, edge_index1 = aug_g1[0], aug_g1[1]
        x2, edge_index2 = aug_g2[0], aug_g2[1]

        z1 = encoder.encode(x1, edge_index1)
        z2 = encoder.encode(x2, edge_index2)

        z1 = encoder.pooling(z1, edge_index1)
        z2 = encoder.pooling(z2, edge_index2)

        sem_loss = self.sem_recon_loss(z1, z2, eta=1.0, bs=bs)
        feat_loss = torch.tensor([0], device=device)
        topo_loss = torch.tensor([0], device=device)
        align_reg = torch.tensor([0], device=device)

        if params['multitask']:
            feat_loss = (self.feat_recon_loss(z1, x, bs=bs) + self.feat_recon_loss(z2, x, bs=bs)) / 2
            topo_loss = (self.topo_recon_loss(z1, edge_index1, ratio=params["topo_recon_ratio"]) +
                         self.topo_recon_loss(z2, edge_index2, ratio=params["topo_recon_ratio"])) / 2

            if params['pareto']:
                # TODO
                pass
            else:
                feat_loss = feat_loss * params['feat_lambda']
                topo_loss = topo_loss * params['topo_lambda']
                sem_loss = sem_loss * params['sem_lambda']

        if params['align_reg_lambda'] > 0:
            z_mean = z1.mean(0)
            align_reg = F.kl_div(z1.log_softmax(dim=-1), z_mean.softmax(dim=-1),
                                 reduction="batchmean") * params['align_reg_lambda']

        losses = {
            'loss': feat_loss + topo_loss + sem_loss + align_reg,
            'feat_loss': feat_loss,
            'topo_loss': topo_loss,
            'sem_loss': sem_loss,
            'align_reg': align_reg,
        }
        return losses
