#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Expert-No-FT
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in cora citeseer pubmed arxiv23 dblp arxiv
do
    for setting in in_context zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data citation --sft_data na \
        --exp_group model-expert-results --no_split --use_params\
        --setting $setting --dataset $dataset
    done
done


for dataset in bookhis bookchild elecomp elephoto sportsfit amazonratings products
do
    for setting in in_context zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data ecommerce --sft_data na \
        --exp_group model-expert-results --no_split --use_params\
        --setting $setting --dataset $dataset
    done
done


for dataset in WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    for setting in in_context zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data kg --sft_data na \
        --exp_group model-expert-results --no_split --use_params\
        --setting $setting --dataset $dataset
    done
done

for dataset in chempcba chemhiv bbbp bace toxcast cyp450 tox21 muv
do
    for setting in in_context zero_shot
    do
        ~/.conda/envs/OFA/bin/python finetune.py --pt_data molecule --sft_data na \
        --exp_group model-expert-results --no_split --use_params\
        --setting $setting --dataset $dataset
    done
done

