#!/bin/bash

cd ../src

gpu=0

sess_per_batch=64    # number of sampled classes
emb_dim=512
batch_size=128
num_negative=5
metric="squaredeuclidean"

max_epochs=20000    # number of iterations
static_epochs=20000
lr=1e-3
keep_prob=0.7
lambda_l2=0.
alpha=0.2


triplet_per_batch=100000
triplet_select="facenet"

name=base_CUB

python base_model_CUB.py --name $name \
    --gpu $gpu --batch_size $batch_size --sess_per_batch $sess_per_batch \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --triplet_select $triplet_select \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --keep_prob $keep_prob \
    --num_negative $num_negative --alpha $alpha

