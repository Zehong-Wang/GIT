import wandb
import os

os.sys.path.append(os.path.abspath(""))

print(os.path.abspath(""))

from finetune import main, main_sweep

dataset = 'arxiv' # [cora, citeseer, pubmed, dblp, arxiv23, arxiv]

sweep_config = {
    "project": "SGFM-Finetune",
    "name": f"Citation Zero-shot Learning Hyper-parameter Tuning -- {dataset}",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/test_mean"},

    "parameters": {
        "setting": {"value": "zero_shot"},
        "pt_lr": {"value": 1e-7},
        "pt_feat_p": {"value": 0.2},
        "pt_edge_p": {"value": 0.2},
        "pt_align_reg_lambda": {"value": 10.0},
        "no_split": {"value": True},

        "pt_data": {"value": "default"},
        "sft_data": {"value": "arxiv"},

        "dataset": {"value": dataset},
        "group": {"value": f"sweep-citation-zero-shot"},

        "sft_epochs": {"min": 10, "max": 500, "q": 10, "distribution": "q_uniform"},
        "sft_lr": {"values": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]},
        "normalize": {"values": ["batch", "none"]}

    },
}

sweep_id = wandb.sweep(sweep=sweep_config)

wandb.agent(sweep_id, function=main_sweep, count=300)
