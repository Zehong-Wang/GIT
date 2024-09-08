#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Hyper-parameter-in-pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1



for dataset in codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    ~/anaconda3/envs/OFA/bin/python ../finetune.py --setting in_context --pretrain_data default --dataset $dataset 
done