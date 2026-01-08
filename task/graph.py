import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils.eval import evaluate, task2metric
from utils.utils import get_device_from_model

EPS = 1e-6


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def sft_graph(model, data, optimizer):
    model.train()
    device = get_device_from_model(model)

    class_node_text_feat = data.dataset.class_node_text_feat
    pos_class_node_text_feat = class_node_text_feat[:len(class_node_text_feat) // 2]
    neg_class_node_text_feat = class_node_text_feat[len(class_node_text_feat) // 2:]

    total_loss = 0

    for sg in data:
        x = sg.node_text_feat.to(device)
        edge_index = sg.edge_index.to(device)
        batch = sg.batch.to(device)

        y_pos = (sg.y == 1).float()
        y_neg = (sg.y == 0).float()
        y_pos = torch.matmul(y_pos, pos_class_node_text_feat) / (y_pos.sum(dim=1) + EPS).view(-1, 1)
        y_neg = torch.matmul(y_neg, neg_class_node_text_feat) / (y_neg.sum(dim=1) + EPS).view(-1, 1)
        y_pos = y_pos.to(device)
        y_neg = y_neg.to(device)
        y = y_pos + y_neg

        z = model.encode_graph(x, edge_index, batch, pool="mean")
        y_pred = model.pooling_lin(z)

        # loss = (F.mse_loss(y_pred, y_pos) + F.mse_loss(y_pred, y_neg)) / 2
        loss = F.mse_loss(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss


def ft_graph(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    setting = params["setting"]
    if setting in ['in_context', 'zero_shot', 'base_zero_shot']:
        return 0

    total_loss = 0
    loader = data[0]

    for sg in loader:
        batch = sg.batch.to(device)
        x = sg.node_text_feat.to(device)
        edge_index = sg.edge_index.to(device)
        y = sg.y.to(device)

        z = model.encode_graph(x, edge_index, batch, pool="mean")
        z = model.pooling_lin(z)
        y_pred = model.classify(z)

        loss = multitask_cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def eval_graph(model, data, split, params):
    model.eval()
    setting = params["setting"]

    if setting in ['base', 'base_zero_shot']:
        train_loader, val_loader, test_loader = data

        # train_value = eval_graph_base(model, train_loader, params)
        train_value = 0  # Does not work for MASSIVE training set
        val_value = eval_graph_base(model, val_loader, params)
        test_value = eval_graph_base(model, test_loader, params)

    elif setting in ['few_shot', 'zero_shot', 'in_context']:
        train_value, val_value, test_value = eval_graph_few_shot(model, data, split, params)

    else:
        raise ValueError(f"Invalid setting: {setting}")

    return {"train": train_value, "val": val_value, "test": test_value, "metric": task2metric[params["task"]]}


def eval_graph_base(model, loader, params):
    device = get_device_from_model(model)

    use_outer_emb = params['setting'] == 'base_zero_shot'
    if use_outer_emb:
        proto_emb = loader.dataset.data.class_node_text_feat.to(device)
        proto_emb = proto_emb[:len(proto_emb) // 2]
        proto_emb = proto_emb.repeat(1, 1)

    y_list, y_pred_list = [], []
    for sg in loader:
        batch = sg.batch.to(device)
        x = sg.node_text_feat.to(device)
        edge_index = sg.edge_index.to(device)
        y = sg.y.to(device)

        z = model.encode_graph(x, edge_index, batch, pool="mean")
        z = model.pooling_lin(z)
        y_pred = model.classify(z) if not use_outer_emb else model.proto_classify(z, proto_emb)

        y_list.append(y.detach())
        y_pred_list.append(y_pred.detach())

    y = torch.cat(y_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    return evaluate(y_pred, y, params=params)


def eval_graph_few_shot(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]

    if isinstance(data, tuple):
        data = data[-1].dataset

    val_as_test = setting in ["zero_shot", "in_context"]
    use_outer_emb = setting in ["zero_shot"]

    train_values, val_values, test_values = [], [], []

    test_idx_s = torch.tensor([idx for task in split['test']['support'] for idx in task.values()]).reshape(-1)
    test_idx_q = torch.tensor([idx for task in split['test']['query'] for idx in task.values()]).reshape(-1)
    test_label_s = torch.tensor([idx for task in split['test']['support_label'] for idx in task.values()]).reshape(
        params['n_task'], -1)
    test_label_q = torch.tensor([idx for task in split['test']['query_label'] for idx in task.values()]).reshape(
        params['n_task'], -1)
    test_idx = np.concatenate([test_idx_s, test_idx_q])

    test_loader = DataLoader(data[test_idx], batch_size=params['bs'], num_workers=1, shuffle=False)

    z_list = []
    for sg in test_loader:
        batch = sg.batch.to(device)
        x = sg.node_text_feat.to(device)
        edge_index = sg.edge_index.to(device)

        z = model.encode_graph(x, edge_index, batch, pool="mean")
        z = model.pooling_lin(z)

        z_list.append(z.detach())
    z = torch.cat(z_list, dim=0)

    s_order = test_label_s[0].sort()[1]
    q_order = test_label_q[0].sort()[1]

    z_s = z[:len(test_idx_s)].reshape(params['n_task'], data.num_classes * 2 * params['n_shot'], -1)
    z_q = z[len(test_idx_s):].reshape(params['n_task'], data.num_classes * 2 * params['n_query'], -1)

    z_s = z_s[:, s_order]
    z_q = z_q[:, q_order]

    if use_outer_emb:
        proto_emb = data.class_node_text_feat.to(device)
        proto_emb = proto_emb.repeat(params['n_task'], 1, 1)
    else:
        z_s = z_s.reshape(params['n_task'], data.num_classes * 2, params['n_shot'], -1)
        proto_emb = z_s.mean(dim=2)

    for task_idx in range(params['n_task']):
        cur_proto_emb = proto_emb[task_idx]
        cur_z_q = z_q[task_idx]

        y_pred = model.proto_classify(cur_z_q, cur_proto_emb)
        y_pred_list = []
        j = 0
        for i in range(data.num_classes * 2 * params['n_query']):
            if i % (params['n_query']) == 0 and i != 0:
                j += 1
            y_pred_list.append(y_pred[i][j])
        y_pred = torch.stack(y_pred_list).sigmoid()
        y = torch.concat([torch.ones(len(cur_z_q) // 2), torch.zeros(len(cur_z_q) // 2)])

        train_value = 0
        test_value = evaluate(y_pred, y, params=params)

        train_values.append(train_value)
        test_values.append(test_value)

        if val_as_test:
            val_values.append(test_value)

    if not val_as_test:
        val_idx_s = torch.tensor([idx for task in split['val']['support'] for idx in task.values()]).reshape(-1)
        val_idx_q = torch.tensor([idx for task in split['val']['query'] for idx in task.values()]).reshape(-1)
        val_label_s = torch.tensor([idx for task in split['val']['support_label'] for idx in task.values()]).reshape(
            params['n_task'], -1)
        val_label_q = torch.tensor([idx for task in split['val']['query_label'] for idx in task.values()]).reshape(
            params['n_task'], -1)
        val_idx = np.concatenate([val_idx_s, val_idx_q])

        val_loader = DataLoader(data[val_idx], batch_size=params['bs'], num_workers=1, shuffle=False)
        z_list = []
        for sg in val_loader:
            batch = sg.batch.to(device)
            x = sg.node_text_feat.to(device)
            edge_index = sg.edge_index.to(device)

            z = model.encode_graph(x, edge_index, batch, pool="mean")
            z = model.pooling_lin(z)

            z_list.append(z.detach())

        z = torch.cat(z_list, dim=0)

        s_order = val_label_s[0].sort()[1]
        q_order = val_label_q[0].sort()[1]

        z_s = z[:len(val_idx_s)].reshape(params['n_task'], data.num_classes * 2 * params['n_shot'], -1)
        z_q = z[len(val_idx_s):].reshape(params['n_task'], data.num_classes * 2 * params['n_query'], -1)

        z_s = z_s[:, s_order]
        z_q = z_q[:, q_order]

        if use_outer_emb:
            proto_emb = data.class_node_text_feat.to(device)
            proto_emb = proto_emb.repeat(params['n_task'], 1, 1)
        else:
            z_s = z_s.reshape(params['n_task'], data.num_classes * 2, params['n_shot'], -1)
            proto_emb = z_s.mean(dim=2)

        for task_idx in range(params['n_task']):
            cur_proto_emb = proto_emb[task_idx]
            cur_z_q = z_q[task_idx]

            y_pred = model.proto_classify(cur_z_q, cur_proto_emb)
            y_pred_list = []
            j = 0
            for i in range(data.num_classes * 2 * params['n_query']):
                if i % (params['n_query']) == 0 and i != 0:
                    j += 1
                y_pred_list.append(y_pred[i][j])
            y_pred = torch.stack(y_pred_list).sigmoid()
            y = torch.concat([torch.ones(len(cur_z_q) // 2), torch.zeros(len(cur_z_q) // 2)])

            val_value = evaluate(y_pred, y, params=params)
            val_values.append(val_value)

    return np.mean(train_values), np.mean(val_values), np.mean(test_values)
