#!/bin/bash

cd ../src

gpu=1
num_threads=2
sess_per_batch=2
batch_size=512
emb_dim=128
feat="sensor"

max_epochs=1100
static_epochs=600
lr=1e-1

name=pretrain_sensor

# python unimodal_pretrain.py --name $name \
#     --gpu $gpu --num_threads $num_threads --batch_size $batch_size \
#     --sess_per_batch $sess_per_batch --max_epochs $max_epochs \
#     --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim \
#     --feat $feat

###############################################################

# model_path="/mnt/work/honda_100h/results/pretrain_sensor_20180401-012104/pretrain_sensor.ckpt-128379"
# network="sae"
# 
# python extract_feature.py --name $name \
#     --gpu $gpu --batch_size 1024 --emb_dim $emb_dim \
#     --feat $feat --network $network --model_path $model_path

###############################################################

feat="sensor_sae"
python unimodal_clustering.py --feat $feat
