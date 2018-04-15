#!/bin/bash

cd ../src

gpu=1

num_threads=2
sess_per_batch=1

num_seg=3
max_epochs=700
static_epochs=350
lr=1e-2
keep_prob=0.5
lambda_l2=0.0

feat="resnet,sensors"    # use comma to separate multiple modalities
emb_dim=128
network="convrtsn"

name=resnet2sensors

python cross_prediction.py --name $name \
    --gpu $gpu --num_threads $num_threads --feat $feat \
    --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --network $network --num_seg $num_seg --keep_prob $keep_prob --lambda_l2 $lambda_l2
