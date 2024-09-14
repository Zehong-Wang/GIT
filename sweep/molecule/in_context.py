import wandb
import os

os.sys.path.append(os.path.abspath(""))

print(os.path.abspath(""))

from finetune import main, main_sweep

dataset = 'toxcast' # ['chempcba', 'chemhiv', 'bbbp', 'bace', 'toxcast', 'cyp450', 'tox21', 'muv']

sweep_config = {
    "project": "SGFM-Finetune",
    "name": f"Molecule In-context Learning Hyper-parameter Tuning -- {dataset}",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "final/test_mean"},

    "parameters": {
        "setting": {"value": "in_context"},
        "pt_lr": {"value": 1e-7},
        "pt_feat_p": {"value": 0.2},
        "pt_edge_p": {"value": 0.2},
        "pt_align_reg_lambda": {"value": 10.0},
        "no_split": {"value": True},

        "pt_data": {"value": "default"},
        "sft_data": {"value": "chempcba"},

        "dataset": {"value": dataset},
        "group": {"value": f"sweep-molecule-in-context"},

        "sft_epochs": {"min": 5, "max": 100, "q": 5, "distribution": "q_uniform"},
        "sft_lr": {"values": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]},
        "normalize": {"values": ["batch", "none"]}

    },
}

sweep_id = wandb.sweep(sweep=sweep_config)

wandb.agent(sweep_id, function=main_sweep, count=100)
