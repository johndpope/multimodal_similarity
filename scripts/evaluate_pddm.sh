#!/bin/bash

cd ../src

model_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'

feat="sensors"
network="rtsn"
num_seg=3
emb_dim=32

gpu=0

python check_inconsistent_pddm.py --model_path $model_path --feat $feat \
                                    --network $network --num_seg $num_seg \
                                    --gpu $gpu --emb_dim $emb_dim
