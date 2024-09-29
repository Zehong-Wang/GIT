#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Single-Model
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in products
do
    for setting in zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --sft_data products --no_split \
        --exp_group specialized-model-results --use_params --setting $setting --dataset $dataset
    done
done