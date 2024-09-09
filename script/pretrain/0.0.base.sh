#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1


~/.conda/envs/OFA/bin/python pretrain.py --dataset --default --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --group base-model