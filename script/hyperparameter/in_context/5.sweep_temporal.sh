#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N In-context-Temporal
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

~/.conda/envs/OFA/bin/python sweep/temporal/in_context.py