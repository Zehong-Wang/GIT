#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Hyper-parameter-in-pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1



for dataset in cora citeseer pubmed arxiv23 dblp
do
    for setting in in_context zero_shot
    do
        ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data na --sft_data na \
        --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --pt_align_reg_lambda 10 \
        --exp_group model-without-pretrain-results --no_split\
        --setting $setting --dataset $dataset
    done
done