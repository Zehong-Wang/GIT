#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for train_ratio in 0.2 0.4 0.6 0.8
do
    ~/anaconda3/envs/OFA/bin/python pretrain.py --dataset default --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --align_reg_lambda 10 --train_ratio $train_ratio --group scaling-law
done