#!/bin/bash


python dataset.py --en $1 --ru $2 --code $3
python tokenize_dataset.py
accelerate launch pretrain.py
python test_results.py

mv results $4
