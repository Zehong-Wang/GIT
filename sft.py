import numpy as np
import warnings
import os
import os.path as osp
import yaml
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling, mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader

from data.finetune_data import get_data
from data.pretrain_data import domain2task, dataset2domain
from model.encoder import Encoder
from utils.utils import seed_everything, load_params, mask2idx, get_scheduler, get_device_from_model, check_path
from utils.args import get_args_sft
from utils.loader import get_sft_loader

from task.node import sft_node
from task.edge import sft_edge
from task.graph import sft_graph

import wandb

warnings.filterwarnings("ignore")


def get_sft(params):
    task = params["task"]

    if task == "node":
        return sft_node
    elif task == "edge":
        return sft_edge
    elif task == "graph":
        return sft_graph
    else:
        raise ValueError("Invalid Task")


get_loader = get_sft_loader


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = get_data(params)
    sft = get_sft(params)

    if params["bs"] != 0:
        data = get_loader(data, params)

    sft_model = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )
    # Load Pretrained Model
    if params["pretrain_dataset"] != 'na':
        template = "lr_{}_hidden_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        path = osp.join(params['pt_model_path'],
                        template.format(params['pt_lr'], params['hidden_dim'], params['backbone'],
                                        params['pt_feat_p'], params['pt_feat_p'],
                                        params['pt_align_reg_lambda'],
                                        params['pretrain_dataset']))
        print("Loader the pretrained encoder model from {}".format(path))

        sft_model = load_params(sft_model, osp.join(path, f'encoder_{params["pt_epochs"]}.pt'))
    sft_model = sft_model.to(device)

    optimizer = AdamW(sft_model.parameters(), lr=params["lr"], weight_decay=params["decay"])

    for epoch in range(1, params['epochs'] + 1):
        sft_loss = sft(model=sft_model, data=data, optimizer=optimizer)
        wandb.log({'loss/sft_loss': sft_loss})

        if params['save']:
            if epoch % 10 == 0:
                template = "lr_{}_hidden_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}_sft_data_{}"
                path = osp.join(params['sft_model_path'],
                                template.format(params['pt_lr'], params['hidden_dim'], params['backbone'],
                                                params['pt_feat_p'], params['pt_feat_p'],
                                                params['pt_align_reg_lambda'],
                                                params['pretrain_dataset'], params['dataset']))
                check_path(path)
                print("Save the instruction fine-tuned model at Epoch {}".format(epoch))
                sft_model.save(osp.join(path, f"encoder_{params['pt_epochs']}_{epoch}.pt"))

    wandb.finish()


if __name__ == "__main__":
    params = get_args_sft()
    params['data_path'] = '/scratch365/zwang43/SGFM/benchmark/cache_data'  # Should be anonymized
    params['pt_model_path'] = "/scratch365/zwang43/SGFM/model/pretrain_model/"  # Should be anonymized
    params['sft_model_path'] = "/scratch365/zwang43/SGFM/model/sft_model/"  # Should be anonymized

    dataset = params["dataset"]
    task = domain2task[dataset2domain[dataset]]
    params['task'] = task
    if task == "graph":
        assert params['bs'] != 0

    if params["use_params"]:
        path = osp.join(os.path.dirname(__file__), 'config', 'sft_param.yaml')
        with open(path, "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[dataset])

    wandb.init(
        project="SGFM-SFT",
        name="Data:{} | PT-Epoch:{}".format(str.upper(params["dataset"]), params["pt_epochs"]),
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        config=params,
        group=params['group'],
    )
    params = dict(wandb.config)
    print(params)

    run(params)
