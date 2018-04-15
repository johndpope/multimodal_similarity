#!/bin/bash

cd ../src

gpu=1

num_threads=2
sess_per_batch=3
num_seg=3
batch_size=256
metric="squaredeuclidean"

max_epochs=750
static_epochs=350
lr=1e-2
keep_prob=1.0
lambda_l2=0.0
lambda_ver=1

triplet_per_batch=200
triplet_select="facenet"
feat="sensors"
emb_dim=32
network="rtsn"
#network="convrtsn"

name=sensors_pretrain_multitask_lambda${lambda_ver}

python multitask_model.py --name $name --lambda_ver $lambda_ver \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size --feat $feat \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob
