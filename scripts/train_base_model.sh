#!/bin/bash

cd ../src

gpu=1

#loss="triplet"
#loss="batch_hard"
loss="lifted"

num_threads=2
sess_per_batch=3
emb_dim=128
#network="tsn"
network="rtsn"
num_seg=3
batch_size=512
metric="squaredeuclidean"

max_epochs=750
static_epochs=350
lr=1e-2
keep_prob=1.0
lambda_l2=0.


if [ "$loss" == "triplet" ]; then

    triplet_per_batch=200
    triplet_select="facenet"
    negative_epochs=0

    #pretrained_model="/mnt/work/honda_100h/results/lr_1e-3_20180326-015741/lr_1e-3.ckpt-27872"

    name=base_model

    #python base_model.py --name $name --pretrained_model $pretrained_model \
    python base_model.py --name $name \
        --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
        --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs \
        --triplet_select $triplet_select --sess_per_batch $sess_per_batch \
        --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
        --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob

elif [ "$loss" == "batch_hard" ]; then

    name=base_model_batchhard_tsn

    python base_model_batchhard.py --name $name \
        --gpu $gpu --num_threads $num_threads --num_seg $num_seg --batch_size $batch_size \
        --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
        --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
        --metric $metric --network $network  --keep_prob $keep_prob #--no_soft

elif [ "$loss" == "lifted" ]; then

    name=base_model_lifted_nosoftplus_margin1.0

    python base_model_lifted.py --name $name \
        --gpu $gpu --num_threads $num_threads --num_seg $num_seg --batch_size $batch_size \
        --sess_per_batch $sess_per_batch --max_epochs $max_epochs --lambda_l2 $lambda_l2\
        --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
        --metric $metric --network $network  --keep_prob $keep_prob --alpha 1.0
fi
