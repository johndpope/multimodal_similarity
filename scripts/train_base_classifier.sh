#!/bin/bash

cd ../src

gpu=1
num_threads=4
event_per_batch=256
max_epochs=150
static_epochs=80
keep_prob=0.7
lr=1e-2

#pretrained_model="/mnt/work/honda_100h/results/lr_1e-3_20180326-015741/lr_1e-3.ckpt-27872"

name=base_classifier

python base_model_classifier.py --name $name \
    --gpu $gpu --num_threads $num_threads --max_epochs $max_epochs \
    --event_per_batch $event_per_batch --keep_prob $keep_prob \
    --learning_rate $lr --static_epochs $static_epochs
