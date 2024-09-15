import wandb
import os

os.sys.path.append(os.path.abspath(""))

print(os.path.abspath(""))

from finetune import main, main_sweep

dataset = 'citeseer' # [cora, citeseer, pubmed, dblp, arxiv23]

sweep_config = {
    "project": "SGFM-Finetune",
    "name": f"Citation FT Learning Hyper-parameter Tuning -- {dataset}",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/test_mean"},

    "parameters": {
        "setting": {"value": "base"},
        "pt_lr": {"value": 1e-7},
        "pt_feat_p": {"value": 0.2},
        "pt_edge_p": {"value": 0.2},
        "pt_align_reg_lambda": {"value": 10.0},

        "pt_data": {"value": "default"},
        "sft_data": {"value": "arxiv"},
        "dataset": {"value": dataset},
        "task": {"value": "node"},
        "group": {"value": f"sweep-citation-base"},

        "sft_epochs": {"value": 100},
        "sft_lr": {"values": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]},

        "normalize": {"values": ["batch", "none"]},
        "lr": {"values": [1e-3, 1e-4, 1e-5, 1e-6]},
        "decay": {"values": [0.0, 1e-6]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config)

wandb.agent(sweep_id, function=main_sweep, count=80)
