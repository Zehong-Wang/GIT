import os
import os.path as osp
import math
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from data.ofa_dataset import MolOFADataset
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, RemoveIsolatedNodes, RandomLinkSplit
from torch_geometric.utils import is_undirected

from utils.utils import idx2mask

citation_datasets = ['arxiv', 'cora', 'citeseer', 'pubmed', 'arxiv23', 'dblp']
ecommerce_datasets = ['bookhis', 'bookchild', 'elecomp', 'elephoto', 'sportsfit', 'amazonratings', 'products']
molecule_datasets = ['chemblpre', 'chempcba', 'chemhiv', 'bbbp', 'bace', 'toxcast', 'cyp450', 'tox21', 'muv']
kg_datasets = ['WN18RR', 'FB15K237', 'codex_s', 'codex_m', 'codex_l', 'NELL995', 'GDELT', 'ICEWS1819']
temporal_datasets = ['Enron', 'Googlemap_CT']

datasets = citation_datasets + ecommerce_datasets + molecule_datasets + kg_datasets + temporal_datasets


def get_data(params):
    dataset_name = params['dataset']
    task = params['task']

    if dataset_name in citation_datasets + ecommerce_datasets:
        if task == 'node':
            return single_graph(params)
        elif task == 'link_pred':
            return single_graph(params)
        else:
            raise ValueError(f"Task {task} not found")

    elif dataset_name in molecule_datasets:
        if task == 'graph':
            return multiple_graphs(params)
        else:
            raise ValueError(f"Task {task} not found")

    elif dataset_name in kg_datasets:
        if task == 'edge':
            return kg_graph(params)
        else:
            raise ValueError(f"Task {task} not found")

    elif dataset_name in temporal_datasets:
        if task == 'edge':
            return temporal_graph(params)
        else:
            raise ValueError(f"Task {task} not found")

    else:
        raise ValueError(f"Dataset {dataset_name} not found")


# Citation networks and e-commerce networks
def single_graph(params):
    data_dir = params['data_path']
    dataset_name = params['dataset']

    assert dataset_name in citation_datasets + ecommerce_datasets

    path = osp.join(data_dir, dataset_name, 'processed', 'geometric_data_processed.pt')

    data = torch.load(path)[0]
    data = ToUndirected()(data)
    data.name = dataset_name

    num_nodes = data.node_text_feat.shape[0]
    num_edges = data.num_edges
    num_classes = data.y.max().item() + 1
    assert num_classes == data.class_node_text_feat.shape[0]
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    data.num_edges = num_edges
    print(f"Dataset: {dataset_name}, #Nodes: {num_nodes}, #Edges: {num_edges}, #Classes: {num_classes}")

    has_mask = hasattr(data, 'train_mask') & hasattr(data, 'val_mask') & hasattr(data, 'test_mask')
    has_masks = hasattr(data, 'train_masks') & hasattr(data, 'val_masks') & hasattr(data, 'test_masks')

    if not has_mask and not has_masks:
        print("No masks found for dataset: ", dataset_name)
    if has_masks:
        len_train_mask = len(data.train_masks)
        len_val_mask = len(data.val_masks)
        len_test_mask = len(data.test_masks)
        assert len_train_mask == len_val_mask
        assert len_train_mask == len_test_mask
        print(len_train_mask, "masks found for dataset: ", dataset_name)
    elif has_mask:
        print("Single mask found for dataset: ", dataset_name)

    return data


def kg_graph(params):
    data_dir = params['data_path']
    dataset_name = params['dataset']

    assert dataset_name in kg_datasets
    path = osp.join(data_dir, dataset_name, 'processed', 'geometric_data_processed.pt')

    data = torch.load(path)[0]
    data.name = dataset_name
    data.raw_texts = None

    num_nodes = data.node_text_feat.shape[0]
    num_edges = data.num_edges
    num_classes = data.y.max().item() + 1
    assert num_classes == data.class_node_text_feat.shape[0]
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    data.num_edges = num_edges

    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    data.test_mask = data.test_mask.bool()

    print(f"Dataset: {dataset_name}, #Nodes: {num_nodes}, #Edges: {num_edges}, #Classes: {num_classes}")

    return data


# Molecule datasets
def multiple_graphs(params):
    data_dir = params['data_path']
    dataset_name = params['dataset']

    assert dataset_name in molecule_datasets

    dataset = MolOFADataset(name=dataset_name, root=data_dir)
    dataset.data.name = dataset_name
    dataset.data.y = dataset.data.y.float()
    num_labels = dataset[0].y.shape[1]
    num_graphs = len(dataset)
    dataset.data.num_classes = num_labels
    print(f"Dataset: {dataset.name}, #Graphs: {num_graphs}, #Labels: {num_labels}")

    if osp.exists(osp.join(data_dir, dataset_name, 'processed', 'groups.pt')):
        dataset.data.groups = torch.load(osp.join(data_dir, dataset_name, 'processed', 'groups.pt'))
    return dataset


# Helper functions for temporal graphs

def split_timestamps(ts, num_splits):
    splits = []
    for i in range(ts.min(), ts.max(), (ts.max() // num_splits) + 1):
        min = i
        if i == ts.max() // num_splits * (num_splits - 1):
            max = ts.max()
        else:
            max = i + ts.max() // num_splits
        splits.append((int(min), int(max)))
    return splits


def get_snapshot(data, start, end, idx, to_undirected=False):
    t = data.t
    name = data.name
    mask = (t >= start) & (t <= end)
    t_range = t[mask]
    edge_index = data.edge_index[:, mask]
    edge_attr = data.edge_attr[mask]

    if to_undirected:
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    y = data.y[mask]

    num_edges = edge_index.shape[1]
    train_ratio = 0.8
    val_ratio = 0.1

    train_split = int(num_edges * train_ratio)
    val_split = int(num_edges * (train_ratio + val_ratio))

    t_order = torch.argsort(t_range)
    train_idx = t_order[:train_split]
    val_idx = t_order[train_split:val_split]
    test_idx = t_order[val_split:]

    train_mask = idx2mask(train_idx, num_edges)
    val_mask = idx2mask(val_idx, num_edges)
    test_mask = idx2mask(test_idx, num_edges)

    print(f"Snapshot: {idx}, #Nodes: {data.x.shape[0]}, #Edges: {num_edges}")

    return Data(node_text_feat=data.x, edge_index=edge_index, edge_text_feat=edge_attr,
                y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                num_classes=data.num_classes, t=t_range, name=f'{name}_{idx}')


def temporal_graph(params):
    data_dir = params['data_path']
    dataset_name = params['dataset']
    snapshot_num = params['snapshot_num']
    snapshot_idx = params['snapshot_idx']

    assert snapshot_idx < snapshot_num
    assert dataset_name in temporal_datasets

    path = osp.join(data_dir, dataset_name)
    edge_index = pd.read_csv(osp.join(path, 'edge_list.csv'), index_col=0)
    e_feat = np.load(osp.join(path, 'e_feat.npy'))
    r_feat = np.load(osp.join(path, 'r_feat.npy'))

    e_feat = torch.Tensor(e_feat)
    r_feat = torch.Tensor(r_feat)

    src = torch.Tensor(edge_index['u'].values).long()
    dst = torch.Tensor(edge_index['i'].values).long()
    rel_idx = torch.Tensor(edge_index['r'].values).long()
    t = torch.Tensor(edge_index['ts'].values).long()
    y = torch.Tensor(edge_index['label'].values).long()

    num_classes = y.max().item() + 1

    data = Data(x=e_feat, edge_index=torch.stack([src, dst], dim=0), edge_attr=r_feat[rel_idx],
                y=y, t=t, name=dataset_name, num_classes=num_classes)

    print(f"Dataset: {dataset_name}, #Snapshot: {snapshot_num}, #Nodes: {e_feat.shape[0]}, #Edges: {src.shape[0]}")

    ts_split = split_timestamps(t, snapshot_num)[snapshot_idx]
    snapshot = get_snapshot(data, ts_split[0], ts_split[1], snapshot_idx)

    return snapshot
