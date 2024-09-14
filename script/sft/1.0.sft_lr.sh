#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N SFT
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

lr=1e-8

for data in arxiv products chempcba FB15K237
do
    ~/.conda/envs/OFA/bin/python sft.py --save --data $data --lr $lr --pt_data default --pt_lr 1e-7 --pt_feat_p 0.2 --pt_edge_p 0.2 --pt_align_reg_lambda 10 --pt_epochs 10 --epochs 500 --group sft-lr
done