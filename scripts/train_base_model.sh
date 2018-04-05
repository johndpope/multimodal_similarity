#!/bin/bash

cd ../src

gpu=0

num_threads=2
sess_per_batch=3
batch_size=256
triplet_per_batch=200
emb_dim=128
triplet_select="facenet"
metric="squaredeuclidean"
network="rtsn"
num_seg=3

max_epochs=1200
static_epochs=600
negative_epochs=0
lr=1e-2
keep_prob=1.0

#pretrained_model="/mnt/work/honda_100h/results/lr_1e-3_20180326-015741/lr_1e-3.ckpt-27872"

name=base_model_rtsn_lr1e-2

#python base_model.py --name $name --pretrained_model $pretrained_model \
python base_model.py --name $name \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob
