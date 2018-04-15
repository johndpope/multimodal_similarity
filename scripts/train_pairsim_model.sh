#!/bin/bash

cd ../src

gpu=1

num_threads=2
num_negative=1     # control the ratio of postive : negative pairs
batch_size=256
sess_per_batch=3
metric="squaredeuclidean"

num_seg=3
max_epochs=1000
static_epochs=400
negative_epochs=500    # when to do hard negative mining
lr=1e-2
keep_prob=1.0
lambda_l2=0.0

feat="sensors"
emb_dim=32
network="rtsn"
#network="convrtsn"

name=sensors_pairsim

python pairsim_model.py --name $name \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size --feat $feat \
    --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --negative_epochs $negative_epochs --num_negative $num_negative --lambda_l2 $lambda_l2
