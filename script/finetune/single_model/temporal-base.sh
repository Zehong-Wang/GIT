#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Single-Model
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1


snapshot_num=10

for dataset in Enron Googlemap_CT
do
    for snapshot_idx in 0 1 2 3 4 5 6 7 8 9
    do
        for setting in base
        do
            ~/.conda/envs/OFA/bin/python finetune.py --pt_data $dataset --sft_data na \
            --use_params --exp_group single-model-results \
            --setting $setting --dataset $dataset \
            --snapshot_idx $snapshot_idx --snapshot_num $snapshot_num
        done
    done
done