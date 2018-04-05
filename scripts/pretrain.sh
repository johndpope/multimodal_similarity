#!/bin/bash

cd ../src

gpu=1
num_threads=2
sess_per_batch=1
batch_size=512
emb_dim=128
feat="sensors"
num_seg=3

max_epochs=1100
static_epochs=600
lr=1e-3
keep_prob=1.0

name=sensor_pretrain_sae

# python unimodal_pretrain_sae.py --name $name \
#     --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
#     --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
#     --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
#     --feat $feat --num_seg $num_seg --keep_prob $keep_prob --no_normalized

###############################################################

model_path="/mnt/work/honda_100h/results/sensor_pretrain_sae_20180404-214133/sensor_pretrain_sae.ckpt-107800"

python unimodal_pretrain_cluster.py --name $name --model_path $model_path \
    --gpu $gpu --batch_size $batch_size --emb_dim $emb_dim --num_seg $num_seg \
    --feat $feat --no_normalized 

###############################################################

# name=sensor_pretrain_pairsim
# model_path="/mnt/work/honda_100h/results/pretrain_sensor_20180401-012104/pretrain_sensor.ckpt-128379"
# network="sae"
# 
# python unimodal_pretrain_pairsim.py --name $name \
#     --gpu $gpu --batch_size $batch_size --emb_dim $emb_dim \
#      --feat $feat --network $network --pretrained_model $model_path
