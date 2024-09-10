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
        for sft_lr in 1e-4 1e-5 1e-6 1e-7 1e-8
        do
            for sft_epochs in 10 50 100
            do
                ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data default --sft_data arxiv --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --pt_align_reg_lambda 10 --exp_group sft-parameter-tuning-sft_lr-sft_epochs-citation --setting $setting --dataset $dataset --sft_lr $sft_lr --sft_epochs $sft_epochs
            done
        done
    done
done