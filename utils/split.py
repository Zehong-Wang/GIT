import json
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils import is_undirected

from data.finetune_data import get_data, citation_datasets, ecommerce_datasets, kg_datasets, molecule_datasets
from utils.utils import idx2mask, mask2idx


def get_split(data, params):
    task = params['task']
    setting = params['setting']

    if setting in ['base', 'base_zero_shot']:
        return base_split(data, params)
    elif setting in ['few_shot', 'in_context', 'zero_shot']:
        return few_shot_split(data, params) if not task == 'graph' else few_shot_split_graph(data, params)
    else:
        raise ValueError(f"Invalid setting: {setting}")


def base_split(data, params):
    task = params['task']
    if task == 'node':
        has_mask = hasattr(data, 'train_mask') & hasattr(data, 'val_mask') & hasattr(data, 'test_mask')
        has_masks = hasattr(data, 'train_masks') & hasattr(data, 'val_masks') & hasattr(data, 'test_masks')
        if has_masks:
            assert len(data.train_masks[0]) == len(data.y)
            train_masks = data.train_masks
            val_masks = data.val_masks
            test_masks = data.test_masks
        elif has_mask:
            assert len(data.train_mask) == len(data.y)
            train_masks = [data.train_mask] * params['repeat']
            val_masks = [data.val_mask] * params['repeat']
            test_masks = [data.test_mask] * params['repeat']
        else:
            raise ValueError("Data does not have masks")

        return [
            {
                'train': train_masks[i],
                'val': val_masks[i],
                'test': test_masks[i]
            } for i in range(len(train_masks))
        ]
    elif task == 'link_pred':
        return [to_link_pred] * params['repeat']
    elif task == 'edge':
        has_mask = hasattr(data, 'train_mask') & hasattr(data, 'val_mask') & hasattr(data, 'test_mask')
        has_masks = hasattr(data, 'train_masks') & hasattr(data, 'val_masks') & hasattr(data, 'test_masks')
        if has_masks:
            assert len(data.train_masks[0]) == len(data.y)
            train_masks = data.train_masks
            val_masks = data.val_masks
            test_masks = data.test_masks
        elif has_mask:
            assert len(data.train_mask) == len(data.y)
            train_masks = [data.train_mask] * params['repeat']
            val_masks = [data.val_mask] * params['repeat']
            test_masks = [data.test_mask] * params['repeat']
        else:
            raise ValueError("Data does not have masks")

        return [
            {
                'train': train_masks[i],
                'val': val_masks[i],
                'test': test_masks[i]
            } for i in range(len(train_masks))
        ]

    elif task == 'graph':
        num_graphs = len(data)
        idx = torch.load(osp.join(params['data_path'], params['dataset'], 'processed', 'data.pt'))[0]
        train_mask = idx2mask(idx['train'], num_graphs)
        val_mask = idx2mask(idx['valid'], num_graphs)
        test_mask = idx2mask(idx['test'], num_graphs)

        return [{
            'train': train_mask,
            'val': val_mask,
            'test': test_mask
        }] * params['repeat']
    else:
        raise ValueError(f"Invalid task: {task}")


# This function works for node classification and edge classification,
# but not yet link prediction and graph classification.
def few_shot_split(data, params):
    splits = base_split(data, params)
    labels = data.y.squeeze()

    no_training = params['setting'] in ['in_context', 'zero_shot']
    val_as_test = params['setting'] in ['in_context', 'zero_shot']
    num_samples = len(labels)

    fs_splits = []
    for split in splits:
        train_idx, val_idx, test_idx = mask2idx(split['train']), mask2idx(split['val']), mask2idx(split['test'])
        train_labels, val_labels, test_labels = data.y[train_idx], data.y[val_idx], data.y[test_idx]

        train_mask_fs = []
        val_mask_s, val_mask_q = [], []
        test_mask_s, test_mask_q = [], []

        ways = get_shared_labels(train_labels, val_labels, test_labels, params['n_train'], params['n_query'])
        print(f"The number of shared labels is {len(ways)}.")

        if not no_training:
            ways_idx = [np.where(labels == way)[0] for way in ways]
            for idx in ways_idx:
                way_idx_train = np.intersect1d(idx, train_idx)
                train_mask_fs.extend(np.random.choice(way_idx_train, params['n_train'], replace=False))
        else:
            print("Does not need to select samples for fine-tuning.")
        train_mask_fs = idx2mask(train_mask_fs, num_samples)

        for task in range(params['n_task']):
            n_way = min(params['n_way'], len(ways))
            rand_ways = np.random.choice(ways, n_way, replace=False)
            way_idx = [np.where(labels == way)[0] for way in rand_ways]

            tmp_idx_s, tmp_idx_q = [], []
            for idx in way_idx:
                way_idx_s = np.intersect1d(idx, train_idx)
                way_idx_q = np.intersect1d(idx, test_idx)
                tmp_idx_s.extend(np.random.choice(way_idx_s, params['n_shot'], replace=False))
                tmp_idx_q.extend(np.random.choice(way_idx_q, params['n_query'], replace=False))
            test_mask_s.append(idx2mask(tmp_idx_s, num_samples))
            test_mask_q.append(idx2mask(tmp_idx_q, num_samples))

        if val_as_test:
            val_mask_q = test_mask_q
            val_mask_s = test_mask_s
        else:
            for task in range(params['n_task']):
                n_way = min(params['n_way'], len(ways))
                rand_ways = np.random.choice(ways, n_way, replace=False)
                way_idx = [np.where(labels == way)[0] for way in rand_ways]

                tmp_idx_s, tmp_idx_q = [], []
                for idx in way_idx:
                    way_idx_s = np.intersect1d(idx, train_idx)
                    way_idx_q = np.intersect1d(idx, val_idx)
                    tmp_idx_s.extend(np.random.choice(way_idx_s, params['n_shot'], replace=False))
                    tmp_idx_q.extend(np.random.choice(way_idx_q, params['n_query'], replace=False))
                val_mask_s.append(idx2mask(tmp_idx_s, num_samples))
                val_mask_q.append(idx2mask(tmp_idx_q, num_samples))

        fs_splits.append({
            'train': train_mask_fs,
            'val': {'support': val_mask_s, 'query': val_mask_q},
            'test': {'support': test_mask_s, 'query': test_mask_q}
        })
    return fs_splits


def few_shot_split_graph(data, params):
    splits = base_split(data, params)
    labels = data.y.view(-1, data.num_classes)  # [num_graphs, num_tasks]

    no_training = params['setting'] in ['in_context', 'zero_shot']
    val_as_test = params['setting'] in ['in_context', 'zero_shot']
    num_samples = len(labels)

    fs_splits = []
    for split in splits:
        # train_labels, val_labels, test_labels = labels[split['train']], labels[split['val']], labels[split['test']]
        train_mask, val_mask, test_mask = split['train'], split['val'], split['test']

        train_idx_fs = []
        val_idx_s, val_idx_q = [], []
        val_label_s, val_label_q = [], []
        test_idx_s, test_idx_q = [], []
        test_label_s, test_label_q = [], []

        if not no_training:
            train_idx_fs, train_label_fs = get_instances_each_task(labels, train_mask, params['n_train'])
            train_idx_fs = [idx for idx in train_idx_fs.values()]
            train_idx_fs = np.concatenate(train_idx_fs)
        else:
            print("Does not need to select samples for fine-tuning.")

        for task in range(params['n_task']):
            cur_idx_s, cur_labels_s = get_instances_each_task(labels, train_mask, params['n_shot'])
            cur_idx_q, cur_labels_q = get_instances_each_task(labels, test_mask, params['n_query'])

            test_idx_s.append(cur_idx_s)
            test_idx_q.append(cur_idx_q)
            test_label_s.append(cur_labels_s)
            test_label_q.append(cur_labels_q)

        if val_as_test:
            val_idx_s = test_idx_s
            val_idx_q = test_idx_q
            val_label_s = test_label_s
            val_label_q = test_label_q
        else:
            for task in range(params['n_task']):
                cur_idx_s, cur_labels_s = get_instances_each_task(labels, train_mask, params['n_shot'])
                cur_idx_q, cur_labels_q = get_instances_each_task(labels, val_mask, params['n_query'])

                val_idx_s.append(cur_idx_s)
                val_idx_q.append(cur_idx_q)
                val_label_s.append(cur_labels_s)
                val_label_q.append(cur_labels_q)

        fs_splits.append({
            "train": train_idx_fs,
            "val": {"support": val_idx_s, "query": val_idx_q, "support_label": val_label_s,
                    "query_label": val_label_q},
            "test": {"support": test_idx_s, "query": test_idx_q, "support_label": test_label_s,
                     "query_label": test_label_q}
        })

    return fs_splits


# Helper functions


def to_link_pred(data):
    if not is_undirected(data.edge_index):
        data = ToUndirected()(data)

    return RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True)(data)


def get_shared_labels(train, val, test, n_shot, n_query):
    base = np.intersect1d(np.unique(train), np.unique(val))
    base = np.intersect1d(base, np.unique(test))

    target = []
    for label in base:
        if (
                torch.sum(train == label) >= n_shot
                and torch.sum(val == label) >= n_query
                and torch.sum(test == label) >= n_query
        ):
            target.append(label)
    return target


def get_instances_each_task(labels, mask, instance_per_task=10):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    labels_full = labels
    labels = labels[mask]
    idx = mask2idx(mask)

    tasks = labels.shape[1]
    idx_pos, idx_neg = [], []

    for task in range(tasks):
        cur_pos = np.where(labels_full[:, task] == 1)[0]
        cur_pos = np.intersect1d(cur_pos, idx)
        idx_pos.append(cur_pos)

        cur_neg = np.where(labels_full[:, task] == 0)[0]
        cur_neg = np.intersect1d(cur_neg, idx)
        idx_neg.append(cur_neg)

    assert len(idx_pos) == len(idx_neg)

    final_idx, final_labels = {}, {}
    for task, (idx_pos, idx_neg) in enumerate(zip(idx_pos, idx_neg)):
        cur_idx, cur_labels = np.array([], dtype=int), np.array([], dtype=int)

        if len(idx_pos) == 0:
            idx_pos = np.array([0] * instance_per_task)
        if len(idx_neg) == 0:
            idx_neg = np.array([0] * instance_per_task)

        idx_pos = np.random.choice(idx_pos, instance_per_task,
                                   replace=False if len(idx_pos) > instance_per_task else True)
        idx_neg = np.random.choice(idx_neg, instance_per_task,
                                   replace=False if len(idx_neg) > instance_per_task else True)

        cur_idx = np.concatenate((cur_idx, idx_pos))
        cur_labels = np.concatenate((cur_labels, np.zeros(len(idx_pos)) + task))
        cur_idx = np.concatenate((cur_idx, idx_neg))
        cur_labels = np.concatenate((cur_labels, tasks * np.ones(len(idx_neg)) + task))

        final_idx[task] = cur_idx
        final_labels[task] = cur_labels

    return final_idx, final_labels
