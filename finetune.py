import numpy as np
import os
import os.path as osp
import shutil
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
from model.finetune_model import TaskModel
from utils.utils import seed_everything, load_params, mask2idx, get_scheduler, get_device_from_model, check_path
from utils.args import get_args_finetune
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.split import get_split
from utils.loader import get_ft_loader

from task.node import ft_node, eval_node, eval_node_few_shot
from task.edge import ft_edge, eval_edge, eval_edge_few_show
from task.link_pred import ft_link_pred, eval_link_pred
from task.graph import ft_graph, eval_graph

import wandb
import warnings

warnings.filterwarnings("ignore")


def get_ft(params):
    task = params["task"]

    if task == "node":
        return ft_node
    elif task == "edge":
        return ft_edge
    elif task == "link_pred":
        return ft_link_pred
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Does not support the task in finetuning.")


def get_eval(params):
    setting = params["setting"]
    task = params["task"]

    if task == "node":
        if setting in ['base', 'base_zero_shot']:
            return eval_node
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_node_few_shot
    elif task == "edge":
        if setting in ['base', 'base_zero_shot']:
            return eval_edge
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_edge_few_show
    elif task == "link_pred":
        if setting in ['base']:
            return eval_link_pred
        elif setting in ['base_zero_shot', 'few_shot', 'zero_shot', 'in_context']:
            raise ValueError("Not support the setting yet in evaluation.")
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Does not support the task in evaluation.")


get_loader = get_ft_loader


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    graph = get_data(params)
    splits = get_split(graph, params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )

    if params["pt_data"] != 'na':
        template = "lr_{}_hidden_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        base_path = params['pt_model_path'] if params["sft_data"] == 'na' else params['sft_model_path']
        path = osp.join(base_path,
                        template.format(params['pt_lr'], params['hidden_dim'], params['backbone'],
                                        params['pt_feat_p'], params['pt_edge_p'], params['pt_align_reg_lambda'],
                                        params['pt_data']))
        if params["sft_data"] != 'na':
            path += "_sft_data_{}".format(params["sft_data"])
        check_path(path)
        print("Load the pretrained model from {}".format(path))

        if params['sft_data'] == 'na':
            encoder_path = osp.join(path, f'encoder_{params["pt_epochs"]}.pt')
        else:
            encoder_path = osp.join(path, f'encoder_{params["pt_epochs"]}_{params["sft_epochs"]}.pt')

        encoder = load_params(encoder, encoder_path)

    model = TaskModel(encoder, num_classes=graph.num_classes)
    model = model.to(device)

    logger = Logger()

    for idx, split in enumerate(splits):
        seed_everything(idx)

        if params["bs"] == 0:
            data = deepcopy(graph)
            if task == 'link_pred':
                data = split(data)
        else:
            # [train_loader, val_loader, test_loader]
            data = get_loader(graph, split, params)

        task_model = deepcopy(model)
        optimizer = AdamW(task_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
        stopper = EarlyStopping(patience=params["early_stop"])

        for epoch in range(1, params["epochs"] + 1):
            loss = finetune(model=task_model, data=data, split=split, optimizer=optimizer, params=params)
            result = evaluate(model=task_model, data=data, split=split, params=params)

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            wandb.log(
                {
                    "train/loss_train": loss,
                    "train/train": result['train'],
                    "train/val": result['val'],
                    "train/test": result['test'],
                    "train/metric": result['metric'],
                }
            )

            if params['save']:
                template = "lr_{}_hidden_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}_sft_data_{}"
                path = osp.join(params["ft_model_path"],
                                template.format(params['lr'], params['hidden_dim'], params['backbone'],
                                                params['feat_p'], params['edge_p'], params['pt_align_reg_lambda'],
                                                params['pt_data'], params['sft_data']),
                                params['dataset'], params['task'], params['setting'])
                check_path(path)
                torch.save(task_model.encoder.state_dict(), osp.join(path, f"encoder_{idx}_{epoch}.pt"))

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })
        if params['save']:
            shutil.copy(osp.join(path, f'encoder_{idx}_{single_best.epoch}.pt', osp.join(path, f'encoder_{idx}.pt')))

    best = logger.get_best()
    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    params = get_args_finetune()
    params['data_path'] = '/scratch365/zwang43/SGFM/benchmark/cache_data'  # Should be anonymized
    params['pt_model_path'] = "/scratch365/zwang43/SGFM/model/pretrain_model/"  # Should be anonymized
    params['sft_model_path'] = "/scratch365/zwang43/SGFM/model/sft_model/"  # Should be anonymized
    params['ft_model_path'] = "/scratch365/zwang43/SGFM/model/finetune_model/"  # Should be anonymized

    dataset = params["dataset"]
    default_task = domain2task[dataset2domain[dataset]]
    if params['task'] is None:
        params['task'] = default_task
    task = params['task']
    if task == "graph":
        assert params['bs'] != 0

    if params["use_params"]:
        with open("ft_param.yaml", "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[task][dataset])

    if params["setting"] in ["base_zero_shot", "zero_shot", "in_context"]:
        params["n_task"] = 500
        params["epochs"] = 1

    tags = [params['task'], params['setting']]
    wandb.init(
        project="SGFM-Finetune",
        name="Data:{} | SFT:{} | PT-Epoch:{}".format(params["dataset"], params["sft_data"], params["pt_epochs"]),
        config=params,
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        group=params['group'],
        tags=tags,
    )
    params = dict(wandb.config)
    print(params)

    run(params)
