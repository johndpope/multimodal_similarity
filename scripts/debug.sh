#!/bin/bash

cd ../src

gpu=0

num_threads=2
sess_per_batch=3
n_h=8
n_w=8
n_C=20
n_input=1536
emb_dim=128
feat="resnet"
network="convrtsn"
num_seg=3
batch_size=512
num_negative=3
metric="squaredeuclidean"

label_num=9
max_epochs=1500
static_epochs=1200
lr=1e-2
keep_prob=0.5
lambda_l2=0.
alpha=0.2


triplet_per_batch=400
triplet_select="facenet"
negative_epochs=0


name=debug_base

python base_model.py --name $name \
    --n_h $n_h --n_w $n_w --n_C $n_C --n_input $n_input \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --num_negative $num_negative --alpha $alpha --feat $feat --label_num $label_num

