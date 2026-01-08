import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from utils.eval import evaluate, task2metric
from utils.utils import get_device_from_model


def predict(z, edge_index=None):
    if edge_index is not None:
        pred = torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))
    else:
        pred = torch.sigmoid(z @ z.t())

    return pred


def ft_link_pred(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    setting = params["setting"]
    if setting in ['base_zero_shot', 'in_context', 'zero_shot']:
        return 0

    data = data[0]
    x = data.node_text_feat.to(device)
    edge_index = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)
    z = model.encode(x, edge_index)
    z = model.pooling_lin(z)

    neg_edge_index = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.size(1))
    edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=1)
    y = torch.cat([data.edge_label, torch.zeros(neg_edge_index.size(1), dtype=torch.long)], dim=0).to(device)

    y_pred = predict(z, edge_label_index)

    loss = F.binary_cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_link_pred(model, data, split, params):
    train_data, val_data, test_data = data

    train_value = eval_link_pred_base(model, train_data, split, params)
    val_value = eval_link_pred_base(model, val_data, split, params)
    test_value = eval_link_pred_base(model, test_data, split, params)

    return {'train': train_value, 'val': val_value, 'test': test_value, 'metric': task2metric[params['task']]}


def eval_link_pred_base(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)

    x = data.node_text_feat.to(device)
    edge_index = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)
    y = data.edge_label.to(device)

    z = model.encode(x, edge_index)
    z = model.pooling_lin(z)

    y_pred = predict(z, edge_label_index)

    return evaluate(y_pred, y, params=params)
