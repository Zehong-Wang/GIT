#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N scaling-law
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1


snapshot_num=10

for dataset in Enron Googlemap_CT
do
    for snapshot_idx in 0 1 2 3 4 5 6 7 8 9
    do
        for train_ratio in 0.2 0.4 0.6 0.8
        do
            for setting in base in_context
            do
                ~/anaconda3/envs/OFA/bin/python finetune.py --pt_data default --sft_data na \
                --exp_group scaling-law --use_params \
                --setting $setting --dataset $dataset --train_ratio $train_ratio \
                --snapshot_idx $snapshot_idx --snapshot_num $snapshot_num
            done
        done
    done
done