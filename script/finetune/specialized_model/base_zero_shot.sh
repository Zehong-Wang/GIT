#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Specialized-Model
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in chemhiv chempcba muv bbbp bace toxcast tox21 cyp450
do
    for setting in base_zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data default --sft_data chemhiv --no_split \
        --exp_group specialized-model-results --use_params --setting $setting --dataset $dataset
    done
done