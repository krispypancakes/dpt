#!/bin/bash

stdbuf -oL -eL python -u src/train_gpt2.py 2>&1 | tee checkpoints/logfile.txt
