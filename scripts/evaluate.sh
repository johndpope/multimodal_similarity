#!/bin/bash

cd ../src


##############################################################################

#model_path='/mnt/work/honda_100h/results/base_model_20180328-230914/base_model.ckpt-35231'     # tsn 256
#model_path='/mnt/work/honda_100h/results/base_model_128_20180329-231615/base_model_128.ckpt-35200'    # tsn 128
#model_path='/mnt/work/honda_100h/results/base_classifier_20180329-075727/base_classifier.ckpt-9450'    # base_classify 256
#model_path='/mnt/work/honda_100h/results/base_model_rtsn_20180404-020419/base_model_rtsn.ckpt-38421'    # rtsn lr=1e-3
#model_path='/mnt/work/honda_100h/results/base_model_rtsn_lr1e-2_20180404-143558/base_model_rtsn_lr1e-2.ckpt-38429'    # rtsn lr=1e-2
#model_path='/mnt/work/honda_100h/results/base_model_rtsn_seg5_20180404-143453/base_model_rtsn_seg5.ckpt-38400'    # rstn lr=1e-3 n_seg=5
#model_path='/mnt/work/honda_100h/results/sensors_pretrain_triplet_20180407-230723/sensors_pretrain_triplet.ckpt-35200'    # sensor tsn
#model_path='/mnt/work/honda_100h/results/base_model_lifted_rtsn_20180408-013733/base_model_lifted_rtsn.ckpt-38400'    # lifted, rstn
#model_path='/mnt/work/honda_100h/results/base_model_lifted_rtsn_20180410-003736/base_model_lifted_rtsn.ckpt-16672'    # lifted, rstn lr=1e-2
#model_path='/mnt/work/honda_100h/results/sensors_pretrain_triplet_20180409-004751/sensors_pretrain_triplet.ckpt-35200'    # sensor rtsn
#model_path='/mnt/work/honda_100h/results/sensor_pretrain_sae_20180404-214133/sensor_pretrain_sae.ckpt-107800'     # sensor encoder-decoder: seq2seqtsn
#model_path='/mnt/work/honda_100h/results/multitask_model_lambda1_20180410-230044/multitask_model_lambda1.ckpt-24016'    # multitask loss, lambda_ver=1
model_path='/mnt/work/honda_100h/results/multimodal_lambdamul0.1_random_20180418-140351/multimodal_lambdamul0.1_random.ckpt-16928'    # multimodal model, lambda_mul=0.1, random selection

feat="resnet"
network="convrtsn"
num_seg=3
emb_dim=128
variable_name="modality_core"

gpu=0

python evaluate_model.py --model_path $model_path --feat $feat \
                   --network $network --num_seg $num_seg \
                   --gpu $gpu --emb_dim $emb_dim --variable_name $variable_name

##############################################################################

# feat="sensor_sae"
# preprocess_func="mean"
# 
# python evaluate.py --feat $feat --preprocess_func $preprocess_func
