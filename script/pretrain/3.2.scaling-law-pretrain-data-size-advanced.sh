#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in scaling_law_4 scaling_law_3 scaling_law_2 scaling_law_1
do
    ~/anaconda3/envs/OFA/bin/python pretrain.py --dataset $dataset --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --align_reg_lambda 10 --group scaling-law
done