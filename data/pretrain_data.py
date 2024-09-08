import os
import os.path as osp
import math
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from data.ofa_dataset import MolOFADataset
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, RemoveIsolatedNodes
from torch_geometric.utils import to_undirected

from data.finetune_data import datasets, citation_datasets, ecommerce_datasets, kg_datasets, molecule_datasets, \
    temporal_datasets, get_data

pretrain_datasets = {
    'default': ['arxiv', 'products', 'WN18RR', 'FB15K237', 'chemblpre', 'chempcba'],
    'citation': citation_datasets,
    'ecommerce': ecommerce_datasets,
    'kg': kg_datasets,
    'molecule': molecule_datasets,
    'arxiv': ['arxiv'],
}
domain2task = {
    'citation': 'node',
    'ecommerce': 'node',
    'kg': 'edge',
    'temporal': 'edge',
    'molecule': 'graph'
}
dataset2domain = {d: 'citation' for d in citation_datasets} | {d: 'ecommerce' for d in ecommerce_datasets} | \
                 {d: 'kg' for d in kg_datasets} | {d: 'molecule' for d in molecule_datasets} | \
                 {d: 'temporal' for d in temporal_datasets}


class VirtualNodeAugmentor:
    def augment(self, data, task):
        assert data.x.ndim == 1, "Node features should be 1D indices"

        if task == 'node':
            return self.add_virtual_nodes_node_classification(data)
        elif task == 'edge':
            return self.add_virtual_nodes_edge_classification(data)
        elif task == 'graph':
            return self.add_virtual_nodes_graph_classification(data)
        else:
            raise ValueError(f"Unknown task: {task}")

    def add_virtual_nodes_node_classification(self, data):
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_nodes) * num_nodes]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes * 2, dtype=torch.long)

        new_edge = torch.tensor([[i, num_nodes + i] for i in range(num_nodes)], dtype=torch.long).t()
        new_edge = to_undirected(new_edge)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_edge_classification(self, data):
        num_edges = data.edge_index.size(1)
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_edges) * num_nodes]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_edges, dtype=torch.long)

        # Note: This is efficient enough
        new_edge = []
        for i in range(num_edges):
            src, dst = data.edge_index[:, i]
            new_edge.append([src, num_nodes + i])
            new_edge.append([num_nodes + i, dst])
        new_edge = torch.tensor(new_edge, dtype=torch.long).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_graph_classification(self, data):
        num_nodes = data.x.shape[0]
        num_node_texts = data.node_text_feat.shape[0]
        node_dim = data.node_text_feat.shape[1]

        groups = data.groups  # the group (i.e. graph) index of each node
        num_groups = groups.max() + 1

        data.x = torch.cat([data.x, torch.ones(num_groups) * num_node_texts]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_groups, dtype=torch.long)

        i_indices = torch.arange(num_nodes, dtype=torch.long)
        new_edge = torch.stack([i_indices, num_nodes + groups], dim=1).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx


def preprocess(data):
    dataset_name = data.name
    if dataset_name in citation_datasets + ecommerce_datasets + kg_datasets:
        data.x = torch.arange(data.num_nodes)

    elif dataset_name in molecule_datasets:
        data = data.data
        data.edge_index = data.pre_edge_index
        data.node_text_feat = data.node_embs

    return data


def postprocess(data):
    keys = ['x', 'edge_index', 'node_text_feat']
    for k, v in data.to_dict().items():
        if k not in keys:
            data[k] = None
    return data


def preprocess_data_dict(data_dict, task_node_idx_dict):
    x_start = 0
    cnt = 0
    for dataset_name, data in data_dict.items():
        task_node_idx = task_node_idx_dict[dataset_name]

        num_nodes = data.x.shape[0]
        num_unique_nodes = data.node_text_feat.shape[0]

        print(f"Preprocessing {dataset_name} with {num_nodes} nodes and {num_unique_nodes} unique nodes")

        data.x += x_start
        x_start += num_unique_nodes

        task_node_idx += cnt
        cnt += num_nodes

        data_dict[dataset_name] = data
        task_node_idx_dict[dataset_name] = task_node_idx

    return data_dict, task_node_idx_dict


def unified_data(params):
    data_path = params['data_path']
    pre_datasets = pretrain_datasets[params['pretrain_dataset']]

    vn = VirtualNodeAugmentor()

    data_dict = {}
    task_node_idx_dict = {}
    for dataset in pre_datasets:
        data = get_data({'data_path': data_path, 'dataset': dataset, 'task': domain2task[dataset2domain[dataset]]})
        data = preprocess(data)
        data, task_node_idx = vn.augment(data, task=domain2task[dataset2domain[dataset]])
        data = postprocess(data)
        data_dict[dataset] = data
        task_node_idx_dict[dataset] = task_node_idx

    data_dict, task_node_idx_dict = preprocess_data_dict(data_dict, task_node_idx_dict)
    unified_dataset = Batch.from_data_list(list(data_dict.values()))

    return unified_dataset, task_node_idx_dict

# if __name__ == '__main__':
#     params = {'data_path': '/scratch365/zwang43/SGFM/benchmark/cache_data', 'pretrain_dataset': 'default'}
#     unified_data, task_node_idx_dict = unified_data(params)
