#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Ablation-SFT-data
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for lr in 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001
do
    for dataset in cora citeseer pubmed arxiv23 dblp
    do
        ~/.conda/envs/OFA/bin/python sft.py --save --data $dataset --lr $lr \
        --pt_data default --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --pt_align_reg_lambda 10 \
        --pt_epochs 10 --epochs 500 --group ablation-sft-data
    done
done