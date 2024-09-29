import os.path as osp
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling, mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader

from data.pretrain_data import unified_data
from model.encoder import Encoder, InnerProductDecoder
from model.pretrain_model import PretrainModel
from utils.utils import seed_everything, get_scheduler, get_device_from_model, check_path
from utils.args import get_args_pretrain
from utils.loader import get_pt_loader

import wandb

get_loader = get_pt_loader


def pretrain(model, loader, optimizer, scheduler=None, **kwargs):
    model.train()
    device = get_device_from_model(model)
    params = kwargs['params']

    for data in loader:
        bs = data.batch_size

        x = data.node_text_feat[data.x].to(device)
        edge_index = data.edge_index.to(device)
        graph = [x, edge_index]

        x1, _ = mask_feature(x, p=params["feat_p"])
        edge_index1, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph1 = [x1, edge_index1]

        x2, _ = mask_feature(x, p=params["feat_p"])
        edge_index2, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph2 = [x2, edge_index2]

        losses = model(graph, aug_graph1, aug_graph2, bs=bs, params=params)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        # model.ema_update_sem_encoder(decay=params["ema"])

        wandb.log(
            {
                "loss/feat_loss": losses["feat_loss"].item(),
                "loss/topo_loss": losses["topo_loss"].item(),
                "loss/sem_loss": losses["sem_loss"].item(),
                "loss/align_reg": losses["align_reg"].item(),
                "loss/loss": loss.item(),
            }
        )


def run(params):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(params["seed"])
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU

    pretrain_data, task_node_idx_dict = unified_data(params)
    train_nodes = torch.concat(list(task_node_idx_dict.values()))
    if params['train_ratio'] != 1:
        train_nodes = torch.tensor(random.sample(train_nodes.tolist(), int(len(train_nodes) * params['train_ratio'])))
    print("Number of training nodes is {}".format(len(train_nodes)))

    encoder = Encoder(
        input_dim=params["input_dim"], hidden_dim=params["hidden_dim"], activation=params["activation"],
        num_layers=params["num_layers"], backbone=params["backbone"], normalize=params["normalize"],
        dropout=params["dropout"]
    )
    feat_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_decoder = InnerProductDecoder(hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"])
    pretrain_model = PretrainModel(encoder=encoder, feat_decoder=feat_decoder, topo_decoder=topo_decoder, ).to(device)

    optimizer = AdamW(pretrain_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["epochs"])

    for i in range(1, params["epochs"] + 1):
        loader = get_loader(pretrain_data, train_nodes, params)
        print("Number of mini-batches is {} at epoch {}.".format(len(loader), i))

        pretrain(model=pretrain_model, loader=loader, optimizer=optimizer, scheduler=scheduler, params=params)

        # Save model
        template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        if params['train_ratio'] != 1:
            template += "_{}".format(params['train_ratio'])

        save_path = osp.join(params['model_path'], template.format(
            params["lr"], params["hidden_dim"], params['num_layers'], params["backbone"],
            params["feat_p"], params["edge_p"], params["align_reg_lambda"], params["pretrain_dataset"]))
        check_path(save_path)

        pretrain_model.save_encoder(osp.join(save_path, f"encoder_{i}.pt"))
        print("Save the model at epoch {}".format(i))

    wandb.finish()


if __name__ == "__main__":
    params = get_args_pretrain()
    # TODO: Anonymize the path
    params['data_path'] = '/scratch365/zwang43/SGFM/benchmark/cache_data'
    params['model_path'] = '/scratch365/zwang43/SGFM/model/pretrain_model'

    wandb.init(
        project="SGFM-Pretrain",
        name="LR:{} | Layers:{} | Fan:{}".format(params["lr"], params["num_layers"], params["fanout"]),
        mode="disabled" if params["debug"] else "online",
        group=params['group'],
        config=params,
    )

    run(params)
