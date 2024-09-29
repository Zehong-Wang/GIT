#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Ablation-SFT-data
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in cora citeseer pubmed arxiv23 dblp arxiv
do
    for sft_data in cora citeseer pubmed arxiv23 dblp arxiv
    do
        for setting in base in_context zero_shot
        do
            ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --sft_data na \
            --exp_group ablation-sft-data --use_params --no_split\
            --setting $setting --dataset $dataset --sft_data $sft_data
        done
    done
done