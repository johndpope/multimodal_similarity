#!/bin/bash

cd ../src


##############################################################################

#model_path='/mnt/work/honda_100h/results/sensors_pretrain_multitask_lambda5_20180411-102739/sensors_pretrain_multitask_lambda5.ckpt-24000'    # lambda_ver=5
model_path='/mnt/work/honda_100h/results/multitask_model_lambda1_20180410-230044/multitask_model_lambda1.ckpt-24016'    # multitask loss, lambda_ver=1

feat="resnet"
network="convrtsn"
num_seg=3
emb_dim=128

gpu=0

python evaluate_pairsim.py --model_path $model_path --feat $feat \
                   --network $network --num_seg $num_seg \
                   --gpu $gpu --emb_dim $emb_dim

##############################################################################

# feat="sensor_sae"
# preprocess_func="mean"
# 
# python evaluate.py --feat $feat --preprocess_func $preprocess_func
