#!/bin/bash

cd ../src


##############################################################################

# #model_path="/mnt/work/honda_100h/results/debug_20180321-001537/debug.ckpt-2002"    # random
# model_path="/mnt/work/honda_100h/results/lr_1e-1_20180327-214732/lr_1e-1.ckpt-23296"    #facenet
# feat="resnet"
# network="tsn"
# num_seg=3
# 
# gpu=0
# 
# python evaluate_model.py --model_path $model_path --feat $feat \
#                    --network $network --num_seg $num_seg \
#                    --gpu $gpu

##############################################################################

feat="sensor_sae"
preprocess_func="mean"

python evaluate.py --feat $feat --preprocess_func $preprocess_func
