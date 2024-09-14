#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Hyper-parameter-in-pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1


snapshot_num=10

for dataset in Enron Googlemap_CT
do
    for snapshot_idx in 0 1 2 3 4 5 6 7 8 9
    do
        for setting in in_context
        do
            ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data default --sft_data na \
            --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --pt_align_reg_lambda 10 \
            --exp_group base-model-results --no_split \
            --setting $setting --dataset $dataset \
            --snapshot_idx $snapshot_idx --snapshot_num $snapshot_num
        done
    done
done