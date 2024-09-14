#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Zero-shot
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

~/.conda/envs/OFA/bin/python sweep/ecommerce/zero_shot.py