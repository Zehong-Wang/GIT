import argparse


def get_args_pretrain():
    parser = argparse.ArgumentParser('Pretrain')

    parser.add_argument("--debug", action="store_true")

    # Base Parameters
    parser.add_argument("--pretrain_dataset", "--pretrain_data", "--dataset", type=str, default="default")
    parser.add_argument("--group", "--exp_group", type=str, default='base')

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--decay", type=float, default=1e-8)
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument('--fanout', type=int, default=10)
    parser.add_argument("--feat_p", type=float, default=0.2)
    parser.add_argument("--edge_p", type=float, default=0.2)
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--use_schedular", type=bool, default=True)

    # Regularizer
    parser.add_argument("--align_reg_lambda", type=float, default=0)

    # Multi-task
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--pareto", action="store_true")
    parser.add_argument("--feat_lambda", type=float, default=1)
    parser.add_argument("--topo_lambda", type=float, default=1)
    parser.add_argument("--topo_recon_ratio", type=float, default=0.1)
    parser.add_argument("--sem_lambda", type=float, default=1)

    args = parser.parse_args()
    return vars(args)


def get_args_sft():
    parser = argparse.ArgumentParser('Instruction Tuning')
    # General Parameters
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--save', action='store_true')
    parser.add_argument("--group", "--exp_group", type=str, default='base')

    # Pre-train Parameters
    parser.add_argument("--pretrain_dataset", "--pretrain_data", "--pt_data", type=str, default="default")
    parser.add_argument("--pt_epochs", type=int, default=10)
    parser.add_argument('--pt_lr', type=float, default=1e-6)
    parser.add_argument('--pt_feat_p', type=float, default=0.2)
    parser.add_argument('--pt_edge_p', type=float, default=0.2)
    parser.add_argument('--pt_align_reg_lambda', type=float, default=0)

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # SFT Parameters
    parser.add_argument("--dataset", "--data", type=str, default='arxiv')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=1e-6)
    parser.add_argument("--bs", type=int, default=0)

    args = parser.parse_args()
    return vars(args)


def get_args_finetune():
    parser = argparse.ArgumentParser('Finetune')
    # General Parameters
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--setting", type=str, default="base",
                        choices=['base', 'few_shot', 'in_context', 'zero_shot', 'base_zero_shot'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument("--group", "--exp_group", type=str, default='base')

    # Few-shot/Zero-shot/In-context Parameters
    parser.add_argument("--n_task", "--n_tasks", type=int, default=20)
    parser.add_argument("--n_way", "--n_ways", type=int, default=5)
    parser.add_argument("--n_train", "--n_trains", type=int, default=10)
    parser.add_argument("--n_shot", "--n_shots", type=int, default=3)
    parser.add_argument("--n_query", "--n_queries", type=int, default=3)

    # Pre-train and SFT Parameters
    parser.add_argument("--pt_data", "--pretrain_dataset", "--pretrain_data", type=str, default="default")
    parser.add_argument("--pt_epochs", type=int, default=10)
    parser.add_argument('--pt_lr', type=float, default=1e-6)
    parser.add_argument('--pt_feat_p', type=float, default=0.2)
    parser.add_argument('--pt_edge_p', type=float, default=0.2)
    parser.add_argument('--pt_align_reg_lambda', type=float, default=0)
    parser.add_argument("--sft_data", "--sft_dataset", type=str, default='na')
    parser.add_argument("--sft_epochs", type=int, default=100)

    # Encoder Parameters
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", "--n_layers", "--layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--backbone", type=str, default="sage")
    parser.add_argument("--normalize", type=str, default="batch")
    parser.add_argument("--dropout", type=float, default=0.15)

    # Downstream Parameters
    parser.add_argument("--dataset", "--data", type=str, default="cora")
    parser.add_argument("--task", type=str, choices=["node", "link_pred", "edge", "graph"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=1e-6)
    parser.add_argument("--bs", "--batch_size", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=200)

    # Temporal Graph Parameters
    parser.add_argument("--snapshot_num", type=int, default=10)
    parser.add_argument("--snapshot_idx", type=int, default=0)

    args = parser.parse_args()
    return vars(args)
