#!/bin/bash

checkpoint_dir=data/checkpoints/dpt/checkpoint-$(date +"%m-%d")
mkdir -p $checkpoint_dir
export PRETRAIN=1 
stdbuf -oL -eL python -u src/dpt.py 2>&1 | tee $checkpoint_dir/logfile.txt

