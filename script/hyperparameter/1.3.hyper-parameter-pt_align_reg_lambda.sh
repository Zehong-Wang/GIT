#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Hyper-parameter-in-pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1



for dataset in cora citeseer pubmed arxiv arxiv23 dblp bookhis bookchild elecomp elephoto sportsfit amazonratings products chempcba chemhiv bbbp bace toxcast cyp450 tox21 muv WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    for setting in in_context zero_shot
    do
        for pt_align_reg_lambda in  0 0.01 0.1 1 10 100
        do
            ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --sft_data na --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --setting $setting --dataset $dataset --pt_align_reg_lambda $pt_align_reg_lambda --exp_group pretrain-parameter-tuning-pt_align_reg_lambda
        done
    done
done