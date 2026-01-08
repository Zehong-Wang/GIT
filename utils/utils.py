import json
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch

from model.encoder import Encoder

EPS = 1e-6


def get_device_from_model(model):
    return next(model.parameters()).device


def get_device(params, optimized_params=None):
    if optimized_params is None or len(optimized_params) == 0:
        device = torch.device(f"cuda:{params['device']}")
    else:
        device = torch.device(f"cuda")
    return device


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def combine_dicts(dicts, decimals=2):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    final_result = {}
    for key, value in result.items():
        if isinstance(value[0], list):
            final_result[key + '_mean'] = np.round(np.mean(value, axis=0), decimals)
            final_result[key + '_std'] = np.round(np.std(value, axis=0), decimals)
        else:
            final_result[key + '_mean'] = np.round(np.mean(value), decimals)
            final_result[key + '_std'] = np.round(np.std(value), decimals)

    return final_result


def mask2idx(mask):
    return torch.where(mask == True)[0]


def idx2mask(idx, num_instances):
    mask = torch.zeros(num_instances, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def load_params(model, path):
    if isinstance(model, Encoder):
        model.load_state_dict(torch.load(path))
    return model


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize(embedding, label=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label, cmap='tab10')
    plt.show()


def sample_proto_instances(labels, split, num_instances_per_class=10):
    y = labels.cpu().numpy()
    target_y = y[split]
    classes = np.unique(target_y)

    class_index = []
    for i in classes:
        c_i = np.where(y == i)[0]
        c_i = np.intersect1d(c_i, split)
        class_index.append(c_i)

    proto_idx = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        proto_idx = np.concatenate((proto_idx, idx[:num_instances_per_class]))

    return proto_idx.astype(int)


def sample_proto_instances_for_graph(labels, split, num_instances_per_class=10):
    y = labels
    ndim = y.ndim
    if ndim == 1:
        y = y.reshape(-1, 1)

    # Map class and instance indices

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    target_y = y[split]
    task_list = target_y.shape[1]

    # class_index_pos = {}
    # class_index_neg = {}
    task_index_pos, task_index_neg = [], []
    for i in range(task_list):
        c_i = np.where(y[:, i] == 1)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_pos.append(c_i)

        c_i = np.where(y[:, i] == 0)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_neg.append(c_i)

    assert len(task_index_pos) == len(task_index_neg)

    # Randomly select instances for each task

    proto_idx, proto_labels = {}, {}
    for task, (idx_pos, idx_neg) in enumerate(zip(task_index_pos, task_index_neg)):
        tmp_proto_idx, tmp_labels = np.array([]), np.array([])

        # Randomly select instance for the task

        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        idx_pos = idx_pos[:num_instances_per_class]
        idx_neg = idx_neg[:num_instances_per_class]

        # Store the randomly selected instances

        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_pos))
        tmp_labels = np.concatenate((tmp_labels, np.ones(len(idx_pos))))
        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_neg))
        tmp_labels = np.concatenate((tmp_labels, np.zeros(len(idx_neg))))

        proto_idx[task] = tmp_proto_idx.astype(int)
        proto_labels[task] = tmp_labels.astype(int)

    return proto_idx, proto_labels
