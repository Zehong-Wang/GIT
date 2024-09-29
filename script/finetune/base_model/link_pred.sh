#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Base-Model
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in cora citeseer pubmed arxiv23 dblp arxiv bookhis bookchild elecomp elephoto sportsfit amazonratings
do
    for setting in base
    do
        ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data default --sft_data na \
        --exp_group base-model-results --use_params --task link_pred \
        --setting $setting --dataset $dataset
    done
done