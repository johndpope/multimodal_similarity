#!/bin/bash

cd ../src

gpu=0
num_threads=2
sess_per_batch=3
batch_size=512
emb_dim=128
feat="sensors"
num_seg=3

max_epochs=1100
static_epochs=600
lr=1e-2
keep_prob=1.0

mode="triplet"
#mode="sae"
#mode="cluster"
#mode="pairsim"

###############################################################

if [ "$mode" = "sae" ]; then

    name=${feat}_pretrain

    python unimodal_pretrain_sae.py --name $name \
        --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
        --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
        --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
        --feat $feat --num_seg $num_seg --keep_prob $keep_prob --no_normalized
fi

###############################################################

if [ "$mode" = "triplet" ]; then

    triplet_per_batch=200
    triplet_select="facenet"
    network="rtsn"

    name=${feat}_pretrain_triplet

    python sensor_model.py --name $name \
        --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
        --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
        --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
        --feat $feat --num_seg $num_seg --keep_prob $keep_prob --triplet_select $triplet_select \
        --triplet_per_batch $triplet_per_batch --network $network
fi

###############################################################

if [ "$mode" = "cluster" ]; then

    name=cluster
    model_path="/mnt/work/honda_100h/results/sensor_pretrain_sae_20180404-214133/sensor_pretrain_sae.ckpt-107800"

    python unimodal_pretrain_cluster.py --name $name --model_path $model_path \
        --gpu $gpu --batch_size $batch_size --emb_dim $emb_dim --num_seg $num_seg \
        --feat $feat --no_normalized 
fi

###############################################################

if [ "$mode" = "pairsim" ]; then

    name="pairsim"
    model_path="/mnt/work/honda_100h/results/sensor_pretrain_sae_20180404-214133/kmeans_20180405-174927"
    
    batch_size=5
    max_epochs=1000
    static_epochs=500
    lr=1e-3
    keep_prob=0.5
    
    python unimodal_pretrain_pairsim.py --name $name \
        --gpu $gpu --batch_size $batch_size --emb_dim $emb_dim \
        --feat $feat --model_path $model_path --learning_rate $lr \
        --max_epochs $max_epochs --static_epochs $static_epochs --keep_prob $keep_prob
fi
