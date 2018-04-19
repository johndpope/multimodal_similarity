#!/bin/bash

cd ../src


##############################################################################

model_path='/mnt/work/honda_100h/results/base_model_rtsn_lr1e-2_20180404-143558/base_model_rtsn_lr1e-2.ckpt-38429'    # rtsn lr=1e-2
sensors_path='/mnt/work/honda_100h/results/resnet2sensors_20180414-154511/resnet2sensors.ckpt-67900'    # resnet2sensors, lambda_l2=0.5

feat="resnet"
network="convrtsn"
num_seg=3
emb_dim=128

gpu=0

python evaluate_late_fusion.py --model_path $model_path --feat $feat \
                   --sensors_path $sensors_path \
                   --network $network --num_seg $num_seg \
                   --gpu $gpu --emb_dim $emb_dim #--use_output
