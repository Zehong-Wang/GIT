#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp

os.sys.path.append(os.path.join(os.path.abspath(""), "../..", "data", "TAG_GFM"))

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from model import Encoder
from utils.utils import (
    load_params,
    check_path,
    get_device_from_model
)

from process_datasets import (
    get_finetune_graph,
    span_node_and_edge_idx,
    filter_unnecessary_attrs
)
from utils.args import get_args
import warnings

warnings.filterwarnings("ignore")

DATASET2TASK = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    "chemhiv": "graph",
    "chempcba": "graph",
}


def preprocess(dataset, splits, task, params):
    dataset = preprocess_dataset(dataset, task)
    splits = preprocess_split(splits, params)
    return dataset, splits


def preprocess_split(splits, params):
    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

    return splits


def preprocess_dataset(dataset, task):
    if task in ['node', 'link']:
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
    elif task == 'graph':
        pass

    return dataset


def encode(encoder, data, task):
    device = get_device_from_model(encoder)
    encoder.eval()

    if task in ['node']:
        data = data.to(device)
        z = encoder(data.node_text_feat, data.edge_index, data.edge_attr)
    elif task in ['link']:
        data = data.to(device)
        z = encoder(data.node_text_feat, data.edge_index, data.edge_attr)
        edge_z = (z[data.edge_index[0]] + z[data.edge_index[1]]) / 2
        z = edge_z
    elif task in ['graph']:
        batch = data
        z_list = []
        for b in batch:
            b = b.to(device)
            z = encoder(b.node_text_feat, b.edge_index, b.edge_attr)
            z = global_mean_pool(z, b.batch)
            z_list.append(z.detach())
        z = torch.cat(z_list, dim=0)
    else:
        raise NotImplementedError('This task is not supported.')

    return z


def run(params):
    all_datasets = ["cora", "pubmed", "arxiv", "wikics", "WN18RR", "FB15K237", "chemhiv", "chempcba"]
    dataset_emb = {}
    dataset_label = {}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU

    for dn in all_datasets:
        task = DATASET2TASK[dn]

        dataset, splits, labels, num_classes, num_tasks = get_finetune_graph(dn)
        dataset, splits = preprocess(dataset, splits, task, params)
        data = dataset[0]
        data.y = labels

        if task == 'graph':
            data = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                num_workers=8,
            )

        encoder = Encoder(
            input_dim=params["input_dim"],
            hidden_dim=params["hidden_dim"],
            activation=params["activation"],
            num_layers=params["num_layers"],
            backbone=params["backbone"],
            normalize=params["normalize"],
            dropout=params["dropout"],
        ).to(device)

        # Load Pretrained Model
        if params["pretrain_dataset"] != 'na':
            # TODO: Anonymize path name in submission version
            prefix = "/scratch365/zwang43/GFM-TAG/model"
            pretrain_task = params['pretrain_task']

            if pretrain_task == 'all':
                path = osp.join(prefix, "pretrain_model",
                                f"lr_{params['pt_lr']}_hidden_{params['hidden_dim']}_backbone_{params['backbone']}"
                                f"_pretrain_{params['pretrain_dataset']}_weight_{params['pt_weight']}"
                                f"_fp_{params['pt_feat_p']}_ep_{params['pt_edge_p']}"
                                f"_alignreg_{params['pt_alignreg_lambda']}_vqreg_{params['pt_vqreg_lambda']}"
                                f"_vqsize_{params['pt_vq_size']}")
            else:
                raise ValueError("Invalid Path")

            encoder = load_params(encoder, osp.join(path, f'encoder_{params["pt_epochs"]}.pt'))
            print("Loader the pretrained encoder model from {}".format(path))

        z = encode(encoder, data, task)

        dataset_label[dn] = labels.detach().cpu()
        dataset_emb[dn] = z.detach().cpu()

    # Save the embeddings and labels
    check_path('embeddings')
    save_path = osp.join('embeddings', 'lr_{}_weight_{}_alignreg_{}_vqreg_{}_vqsizse_{}.pt'.format(
        params['pt_lr'], params['pt_weight'], params['pt_alignreg_lambda'], params['pt_vqreg_lambda'],
        params['pt_vq_size']))
    torch.save({'embeddings': dataset_emb, 'labels': dataset_label}, save_path)

def main():
    params = get_args()
    run(params)


if __name__ == "__main__":
    main()
