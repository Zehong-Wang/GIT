#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for hidden_dim in 128 512 1024 2048
do
    ~/anaconda3/envs/OFA/bin/python pretrain.py --dataset default --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --align_reg_lambda 10 --hidden_dim $hidden_dim --group scaling-law
done