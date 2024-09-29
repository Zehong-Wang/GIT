#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N scaling-law
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1


for dataset in cora citeseer pubmed arxiv23 dblp arxiv bookhis bookchild elecomp elephoto sportsfit amazonratings products WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819
do
    for pt_data in scaling_law_4
    do
        for setting in in_context zero_shot
        do
            ~/.conda/envs/OFA/bin/python finetune.py --sft_data na \
            --exp_group scaling-law --use_params \
            --setting $setting --dataset $dataset --pt_data $pt_data
        done
    done
done


for dataset in chemhiv bbbp bace toxcast cyp450 tox21 muv chempcba
do
    for pt_data in scaling_law_4
    do
        for setting in in_context base_zero_shot
        do
            ~/.conda/envs/OFA/bin/python finetune.py --sft_data na \
            --exp_group scaling-law --use_params \
            --setting $setting --dataset $dataset --pt_data $pt_data
        done
    done
done


for dataset in cora citeseer pubmed arxiv23 dblp arxiv bookhis bookchild elecomp elephoto sportsfit amazonratings WN18RR FB15K237 codex_s codex_m codex_l NELL995 GDELT ICEWS1819 chemhiv bbbp bace toxcast cyp450 tox21 muv products chempcba
do
    for pt_data in scaling_law_4
    do
        for setting in base
        do
            ~/.conda/envs/OFA/bin/python finetune.py --sft_data na \
            --exp_group scaling-law --use_params \
            --setting $setting --dataset $dataset --pt_data $pt_data
        done
    done
done