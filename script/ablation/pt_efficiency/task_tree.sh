#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for bs in 512 1024 2048 4096 8192 16384 32768 65536
do
    ~/.conda/envs/OFA/bin/python pretrain.py --dataset default --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --align_reg_lambda 10 --group ablation-efficiency --bs $bs --epochs 1
done