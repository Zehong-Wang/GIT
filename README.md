# GIT: Graph Generality Identifier on Task-Trees

[![ICML 2025](https://img.shields.io/badge/ICML-2025-blue.svg)](https://openreview.net/forum?id=BSqf2k01ag)
[![arXiv](https://img.shields.io/badge/arXiv-2412.16441-b31b1b.svg)](https://arxiv.org/abs/2412.16441)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees"** accepted at ICML 2025.

**Authors:** Zehong Wang, Zheyuan Zhang, Tianyi Ma, Nitesh V Chawla, Chuxu Zhang, Yanfang Ye


## 🌟 Overview

Graph-structured data is everywhere---from social networks to molecular structures---but building general-purpose models for graphs has been difficult due to the wide variety of graph types and tasks. Inspired by the success of foundation models in text and vision, this work introduces a new approach to generalize across different graph tasks using a concept called **"task-trees."**

### Key Contributions

- **Task-Trees**: A unified structure that captures the essential parts of a graph relevant to a specific task and unifies different types of graph tasks (node, edge, graph-level) into a common format.
- **Theoretical Foundation**: We provide theoretical analysis on the stability, transferability, and generalization properties of task-trees.
- **GIT Model**: A graph foundation model pretrained on task-trees from diverse graphs, demonstrating strong performance across 30+ datasets in five domains.
- **Multiple Learning Paradigms**: Support for fine-tuning, few-shot learning, in-context learning, and zero-shot generalization.

## 🏗️ Architecture

GIT employs a three-stage training paradigm:

1. **Pretraining**: Learning generalizable patterns from diverse graphs via task-tree reconstruction
2. **Supervised Fine-Tuning (SFT)**: Domain-specific adaptation with supervised data
3. **Task Fine-Tuning**: Final adaptation to downstream tasks with minimal data

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.1.0+
- PyTorch Geometric 2.5.3
- CUDA 11.8 (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/GIT.git
cd GIT

# Create a virtual environment (optional but recommended)
conda env create -f environment.yml
conda activate GIT
```

## 🚀 Quick Start

### 1. Data Preparation

TODO: Add the link to the data.

Download the data from [Google Drive]() and put it in the `cache_data/` directory.

### 2. Pretraining

Train a foundation model from scratch on diverse graphs:

```bash
# Basic pretraining command
python pretrain.py \
    --pretrain_dataset default \
    --lr 0.0000001 \
    --epochs 10 \
    --feat_p 0.2 \
    --edge_p 0.2 \
    --align_reg_lambda 10.0 \
    --hidden_dim 512 \
    --num_layers 4 \
    --backbone gcn \
    --seed 42
```

**Key Hyperparameters:**
- `--pretrain_dataset`: Pretraining data source
- `--lr`: Learning rate for pretraining
- `--epochs`: Number of pretraining epochs
- `--feat_p`: Feature masking probability
- `--edge_p`: Edge dropout probability
- `--align_reg_lambda`: Weight for alignment regularization
- `--hidden_dim`: Hidden dimension of GNN encoder
- `--num_layers`: Number of GNN layers
- `--backbone`: GNN backbone architecture (gcn, gat, gin, sage)

### 3. Supervised Fine-Tuning (Optional)

Perform domain-specific adaptation:

```bash
# SFT on citation domain
python sft.py \
    --dataset cora \
    --pretrain_dataset default \
    --use_params
```

### 4. Downstream Task Adaptation

#### Fine-tuning (Standard Setting)

```bash
# Fine-tune on node classification
python finetune.py \
    --dataset cora \
    --setting base \
    --pt_data default \
    --use_params
```

#### Few-Shot Learning

```bash
# Few-shot learning (1-shot for Cora, 5-shot for Arxiv)
python finetune.py \
    --dataset cora \
    --setting few_shot \
    --pt_data default \
    --n_train 1 \
    --n_way 5 \
    --n_shot 1 \
    --n_query 15 \
    --n_task 500 \
    --use_params
```

#### Zero-Shot Generalization

```bash
# Zero-shot transfer without any training
python finetune.py \
    --dataset cora \
    --setting zero_shot \
    --pt_data default \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --n_task 500 \
    --use_params
```

#### In-Context Learning

```bash
# In-context learning with examples in the context
python finetune.py \
    --dataset cora \
    --setting in_context \
    --pt_data default \
    --n_way 5 \
    --n_shot 5 \
    --n_query 15 \
    --n_task 500 \
    --use_params
```

## 📊 Supported Datasets

GIT supports over 30 datasets across 5 domains:

### Citation Networks
- **Node Classification**: Cora, CiteSeer, PubMed, DBLP, Arxiv, Arxiv23
- **Link Prediction**: Cora, CiteSeer, PubMed, DBLP, Arxiv, Arxiv23

### E-commerce Networks  
- **Node Classification**: BookHis, BookChild, EleComp, ElePhoto, SportsFit, AmazonRatings, Products
- **Link Prediction**: BookHis, BookChild, EleComp, ElePhoto, SportsFit, AmazonRatings

### Knowledge Graphs
- **Edge Classification**: WN18RR, FB15K237, NELL995, CODEX-S, CODEX-M, CODEX-L, GDELT, ICEWS18-19, Enron, Googlemap_CT

### Molecular Graphs
- **Graph Classification**: BBBP, BACE, ToxCast, Tox21, CYP450, ChemHIV, MUV, ChemPCBA


## 📝 Logging and Monitoring

We use [Weights & Biases](https://wandb.ai/) for experiment tracking. To enable logging:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Run experiments with `--debug` flag removed (or set to `False`)

To disable wandb logging, add `--debug` flag to your command.

## 🤝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025towards,
    title={Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees},
    author={Zehong Wang and Zheyuan Zhang and Tianyi Ma and Nitesh V Chawla and Chuxu Zhang and Yanfang Ye},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=BSqf2k01ag}
}
```

## 📧 Contact

For questions or feedback, please contact:

- Zehong Wang: [zwang43@nd.edu](mailto:zwang43@nd.edu)
- Open an issue on GitHub

## 🙏 Acknowledgments

This repository is built upon the following excellent codebases:

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): Graph neural network library
- [OGB](https://ogb.stanford.edu/): Open Graph Benchmark
- [OFA](https://github.com/LechengKong/OneForAll): One for All graph pretraining framework

We thank the authors for their great work and open-sourcing their code!
