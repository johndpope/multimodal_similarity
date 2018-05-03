#!/bin/bash

cd ../src

gpu=1

loss="triplet"

num_threads=2
sess_per_batch=3
emb_dim=32
n_h=150
n_w=240
n_C=8
n_input=2
feat="segment"
network="convrtsn"
num_seg=3
batch_size=512
num_negative=3
metric="squaredeuclidean"

label_num=93
max_epochs=750
static_epochs=500
lr=1e-2
keep_prob=1.0
lambda_l2=0.
alpha=0.2


triplet_per_batch=400
triplet_select="facenet"
negative_epochs=0

#pretrained_model="/mnt/work/honda_100h/results/lr_1e-3_20180326-015741/lr_1e-3.ckpt-27872"

name=PDDM_${feat}_${label_num}

#python base_model.py --name $name --pretrained_model $pretrained_model \
python pddm_model.py --name $name \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
    --n_h $n_h --n_w $n_w --n_C $n_C --n_input $n_input \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --num_negative $num_negative --alpha $alpha --feat $feat --label_num $label_num

