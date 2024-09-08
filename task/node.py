import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.eval import evaluate, task2metric
from utils.utils import get_device_from_model


def sft_node(model, data, optimizer):
    model.train()
    device = get_device_from_model(model)

    is_loader = not isinstance(data, Data)

    if not is_loader:
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        y = data.class_node_text_feat[data.y.squeeze()].to(device)

        z = model.encode(x, edge_index)
        y_pred = model.pooling_lin(z)

        loss = F.mse_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    else:
        total_loss = 0
        for sg in data:
            bs = sg.batch_size

            x = sg.node_text_feat.to(device)
            edge_index = sg.edge_index.to(device)
            y = sg.class_node_text_feat[sg.y.squeeze()][:bs].to(device)

            z = model.encode(x, edge_index)[:bs]
            y_pred = model.pooling_lin(z)

            loss = F.mse_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss


def ft_node(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    setting = params["setting"]
    if setting in ['base_zero_shot', 'in_context', 'zero_shot']:
        return 0

    is_loader = not isinstance(data, Data)

    if not is_loader:
        train_mask = split["train"]
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y[train_mask].squeeze().to(device)

        z = model.encode(x, edge_index)[train_mask]
        z = model.pooling_lin(z)
        y_pred = model.classify(z)

        loss = F.cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    else:
        total_loss = 0
        loader = data[0]
        for sg in loader:
            bs = sg.batch_size

            x = sg.node_text_feat.to(device)
            edge_index = sg.edge_index.to(device)
            y = sg.y[:bs].squeeze().to(device)

            z = model.encode(x, edge_index)[:bs]
            z = model.pooling_lin(z)
            y_pred = model.classify(z)

            loss = F.cross_entropy(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss


def eval_node(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)

    train_mask, val_mask, test_mask = split["train"], split["val"], split["test"]

    is_loader = not isinstance(data, Data)
    use_outer_emb = params['setting'] in ['base_zero_shot']

    if use_outer_emb:
        proto_emb = data.class_node_text_feat.to(device)

    if not is_loader:
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.squeeze().to(device)

        z = model.encode(x, edge_index)
        z = model.pooling_lin(z)
        if not use_outer_emb:
            y_pred = model.classify(z)
        else:
            y_pred = model.proto_classify(z, proto_emb)
    else:
        y_list, y_pred_list = [], []
        loader = data[-1]
        for sg in loader:
            bs = sg.batch_size

            x = sg.node_text_feat.to(device)
            edge_index = sg.edge_index.to(device)
            y = sg.y[:bs].squeeze().to(device)

            z = model.encode(x, edge_index)[:bs]
            z = model.pooling_lin(z)
            if not use_outer_emb:
                y_pred = model.classify(z)
            else:
                y_pred = model.proto_classify(z, proto_emb)

            y_list.append(y.detach())
            y_pred_list.append(y_pred.detach())
        y = torch.cat(y_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

    train_value = evaluate(y_pred, y, train_mask, params)
    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {'train': train_value, 'val': val_value, 'test': test_value, 'metric': task2metric[params['task']]}


def eval_node_few_shot(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]

    # Get node embedding and labels
    is_loader = not isinstance(data, Data)

    if not is_loader:
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.squeeze().to(device)

        z = model.encode(x, edge_index)
        z = model.pooling_lin(z)
    else:
        y_list, z_list = [], []
        for sg in data[-1]:
            bs = sg.batch_size

            x = sg.node_text_feat.to(device)
            edge_index = sg.edge_index.to(device)
            y = sg.y[:bs].squeeze().to(device)

            z = model.encode(x, edge_index)[:bs]
            z = model.pooling_lin(z)

            y_list.append(y.detach())
            z_list.append(z.detach())
        y = torch.cat(y_list, dim=0)
        z = torch.cat(z_list, dim=0)

    val_as_test = setting in ["zero_shot", "in_context"]
    use_outer_emb = setting in ["zero_shot"]
    num_classes = y.max().item() + 1

    train_values, val_values, test_values = [], [], []

    # Validation
    n_task = len(split["val"]["support"])
    for i in range(n_task):
        s_mask = split["val"]["support"][i]
        q_mask = split["val"]["query"][i]

        z_q, y_q, z_s, y_s = z[q_mask], y[q_mask], z[s_mask], y[s_mask]
        if use_outer_emb:
            proto_emb = data.class_node_text_feat.to(device)
        else:
            proto_emb = model.get_class_prototypes(z_s, y_s, num_classes).detach()

        pred = model.proto_classify(z_q, proto_emb)

        train_value = 0
        val_value = evaluate(pred, y_q, params=params)

        train_values.append(train_value)
        val_values.append(val_value)
        if val_as_test:
            test_values.append(val_value)

    # Test
    if not val_as_test:
        n_task = len(split["test"]["support"])
        for i in range(n_task):
            s_mask = split["test"]["support"][i]
            q_mask = split["test"]["query"][i]

            z_q, y_q, z_s, y_s = z[q_mask], y[q_mask], z[s_mask], y[s_mask]
            if use_outer_emb:
                proto_emb = data.class_node_text_feat.to(device)
            else:
                proto_emb = model.get_class_prototypes(z_s, y_s, num_classes).detach()

            pred = model.proto_classify(z_q, proto_emb)

            test_values.append(evaluate(pred, y_q, params=params))

    return {'train': np.mean(train_values),
            'val': np.mean(val_values),
            'test': np.mean(test_values),
            'metric': task2metric[params['task']]}
