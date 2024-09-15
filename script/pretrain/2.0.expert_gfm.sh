#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Pretrain
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in citation ecommerce molecule kg arxiv cora citeseer pubmed arxiv23 dblp bookhis bookchild elecomp elephoto sportsfit amazonratings products chemblpre chempcba chemhiv bbbp bace toxcast cyp450 tox21 muv WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    ~/.conda/envs/OFA/bin/python pretrain.py --dataset $dataset --fanout 10 --num_layers 2 --lr 1e-7 --edge_p 0.2 --feat_p 0.2 --align_reg_lambda 10 --group expert_gfm
done