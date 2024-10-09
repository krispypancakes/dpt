#!/bin/bash

checkpoint_dir=checkpoints/checkpoint-$(date +"%m-%d")
mkdir -p $checkpoint_dir

stdbuf -oL -eL python -u src/train_gpt2.py 2>&1 | tee $checkpoint_dir/logfile.txt
