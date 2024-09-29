#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Specialized-Model
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
            ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --sft_data products \
            --use_params --exp_group specialized-model-results --no_split \
            --setting $setting --dataset $dataset \
            --snapshot_idx $snapshot_idx --snapshot_num $snapshot_num
        done
    done
done