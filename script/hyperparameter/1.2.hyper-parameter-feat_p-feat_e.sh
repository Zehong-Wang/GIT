#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Hyper-parameter-in-pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1



for dataset in cora citeseer pubmed arxiv arxiv23 dblp bookhis bookchild elecomp elephoto sportsfit amazonratings products chemhiv bbbp bace cyp450 tox21 muv WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    for setting in zero_shot
    do
        for pt_feat_p in 0.1 0.2
        do
            for pt_edge_p in 0.1 0.2
            do
                ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --setting $setting --dataset $dataset --sft_data na --pt_lr 1e-7 --pt_epochs 10 --pt_edge_p $pt_edge_p --pt_feat_p $pt_feat_p --exp_group pretrain-parameter-tuning-pt_feat_p-pt_edge_p
            done
        done
    done
done