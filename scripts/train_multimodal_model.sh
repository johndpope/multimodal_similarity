#!/bin/bash

cd ../src

gpu=1

event_per_batch=1000
sess_per_batch=3
num_negative=5
num_seg=3
batch_size=512
metric="squaredeuclidean"

label_num=9
max_epochs=2000
static_epochs=1000
multimodal_epochs=0
lr=1e-2
keep_prob=0.5
lambda_l2=0.0
lambda_multimodal=0.1

triplet_per_batch=200
multimodal_select="random"
alpha=0.2
feat="resnet,sensors,segment"
emb_dim=128
network="convrtsn"
optimizer="ADAM"

name=multimodal_full_lambdamul${lambda_multimodal}_labelnum${label_num}_0.3
#name=debug

segment_path='/mnt/work/honda_100h/results/PDDM_segment_labelnum9_20180509-164911/PDDM_segment_labelnum9.ckpt-4500'    # PDDM segment, label_num=9
sensors_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum9_20180509-164952/PDDM_sensors_labelnum9.ckpt-4500'    # PDDM sensors, label_num=9

#python multimodal_model_hardonly.py --name $name --lambda_multimodal $lambda_multimodal \
python multimodal_model.py --name $name --lambda_multimodal $lambda_multimodal \
    --gpu $gpu --batch_size $batch_size --feat $feat --multimodal_epochs $multimodal_epochs \
    --triplet_per_batch $triplet_per_batch --max_epochs $max_epochs --num_negative $num_negative \
    --sess_per_batch $sess_per_batch --lambda_l2 $lambda_l2 --label_num $label_num \
    --learning_rate $lr --static_epochs $static_epochs --emb_dim $emb_dim --alpha $alpha \
    --metric $metric --network $network --num_seg $num_seg --keep_prob $keep_prob \
    --multimodal_select $multimodal_select --optimizer $optimizer --event_per_batch $event_per_batch \
    --sensors_path $sensors_path --segment_path $segment_path --no_joint #--weighted

