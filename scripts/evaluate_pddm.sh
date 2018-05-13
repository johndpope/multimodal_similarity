#!/bin/bash

cd ../src

#model_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'
model_path='/mnt/work/honda_100h/results/PDDM_segment_93_20180505-225524/PDDM_segment_93.ckpt-46500'    # PDDM segmentation

feat="segment"
network="rtsn"
n_input=357
num_seg=3
emb_dim=32

gpu=0

python check_inconsistent_pddm.py --model_path $model_path --feat $feat \
                                    --network $network --num_seg $num_seg \
                                    --gpu $gpu --emb_dim $emb_dim --n_input $n_input
