#!/bin/bash

cd ../src

gpu=0

num_threads=2
sess_per_batch=3
num_negative=3
num_seg=3
batch_size=256
metric="squaredeuclidean"

max_epochs=750
static_epochs=350
lr=1e-2
keep_prob=1.0
lambda_l2=0.0
lambda_multimodal=1

triplet_per_batch=200
triplet_select="facenet"
alpha=0.2
feat="resnet,sensors"
emb_dim=128
network="convrtsn"
optimizer="ADAM"

name=debug_hallucination

sensors_path='/mnt/work/honda_100h/results/base_model_sensors_20180419-231909/base_model_sensors.ckpt-24000'

python modality_hallucination.py --name $name --lambda_multimodal $lambda_multimodal \
    --gpu $gpu --num_threads $num_threads --batch_size $batch_size --feat $feat \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs --num_negative $num_negative \
    --triplet_select $triplet_select --sess_per_batch $sess_per_batch --lambda_l2 $lambda_l2 \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim --alpha $alpha \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --sensors_path $sensors_path --optimizer $optimizer 

