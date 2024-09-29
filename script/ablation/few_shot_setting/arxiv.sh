#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Ablation-SFT-Other-Domains
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

sft_data=arxiv
dataset=arxiv

for sft_data in na arxiv
do
    for n_shot in 1 3 5
    do
        for n_way in 3 5
        do
            ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data default --sft_data $sft_data \
            --exp_group ablation-few-shot-setting --use_params --no_split\
            --setting in_context --dataset $dataset --n_shot $n_shot --n_way $n_way
        done
    done
done
