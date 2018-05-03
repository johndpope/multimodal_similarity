#!/bin/bash

cd ../src


##############################################################################


feat="resnet"
#feat="sensors"
network="convrtsn"
#network="rtsn"
num_seg=3
emb_dim=128
#emb_dim=32
variable_name="modality_core/"

gpu=1

#model_path='/mnt/work/honda_100h/results/base_model_sensors_20180426-103049/base_model_sensors.ckpt-31000'    # sensors base model
#model_path='/mnt/work/honda_100h/results/base_model_sensors_alpha1_20180426-162327/base_model_sensors_alpha1.ckpt-31000'    # sensors, alpha=1
#model_path='/mnt/work/honda_100h/results/base_model_convtsn_20180426-103237/base_model_convtsn.ckpt-23250'    # base_model convtsn
#model_path='/mnt/work/honda_100h/results/base_model_convrtsn_20180426-103711/base_model_convrtsn.ckpt-23263'    # base_model convrtsn
#model_path='/mnt/work/honda_100h/results/base_model_convbirtsn_20180426-103848/base_model_convbirtsn.ckpt-23259'    # base_model convbirtsn
#model_path='/mnt/work/honda_100h/results/base_model_dropout0.7_20180427-120849/base_model_dropout0.7.ckpt-23261'    # base_model convrtsn dropout=0.7
#model_path='/mnt/work/honda_100h/results/base_model_dropout0.5_20180428-112516/base_model_dropout0.5.ckpt-23277'    # base_model convrtsn dropout=0.5
#model_path='/mnt/work/honda_100h/results/PDDM_sensors_labelnum93_full_20180502-085626/PDDM_sensors_labelnum93_full.ckpt-46500'    # PDDM sensors
#model_path='/mnt/work/honda_100h/results/base_model_labelnum75_20180429-114115/base_model_labelnum75.ckpt-18774'    # base_model label_num=75
#model_path='/mnt/work/honda_100h/results/base_model_labelnum57_20180429-114154/base_model_labelnum57.ckpt-14250'    # label_num=57
#model_path='/mnt/work/honda_100h/results/base_model_labelnum39_20180429-114220/base_model_labelnum39.ckpt-9750'    # label_num=39
#model_path='/mnt/work/honda_100h/results/base_model_labelnum21_20180429-114340/base_model_labelnum21.ckpt-5250'    # label_num=21
#model_path='/mnt/work/honda_100h/results/base_model_labelnum3_20180429-114358/base_model_labelnum3.ckpt-750'    # label_num=3
#model_path='/mnt/work/honda_100h/results/base_model_labelnum3_20180427-231939/base_model_labelnum3.ckpt-750'    # label_num3 without dropout
model_path='/mnt/work/honda_100h/results/base_model_labelnum63_20180427-121127/base_model_labelnum63.ckpt-15758'    #label_num=63 without dropout

#python evaluate_hallucination.py --model_path $model_path --feat $feat \
python evaluate_model.py --model_path $model_path --feat $feat \
                   --network $network --num_seg $num_seg \
                   --gpu $gpu --emb_dim $emb_dim #--variable_name $variable_name

##############################################################################

# feat="sensor_sae"
# preprocess_func="mean"
# 
# python evaluate.py --feat $feat --preprocess_func $preprocess_func

############ old models ###########################

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
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul0.1_random_20180418-140351/multimodal_lambdamul0.1_random.ckpt-16928'    # multimodal model, lambda_mul=0.1, random selection
#model_path='/mnt/work/honda_100h/results/multimodal_base400_20180420-083408/multimodal_base400.ckpt-24004'    # base_model with triplet_per_batch=400
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul1_epochs150_nojoint_20180420-083225/multimodal_lambdamul1_epochs150_nojoint.ckpt-24010'    # nojoint
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul1_epochs150_20180420-011209/multimodal_lambdamul1_epochs150.ckpt-24028'    # mutlimodal lambda_mul=1, mul_epochs=150
#model_path='/mnt/work/honda_100h/results/base_model_sensors_20180419-231909/base_model_sensors.ckpt-24000'    # sensors pretrain
#model_path='/mnt/work/honda_100h/results/debug_hallucination_20180420-001554/debug_hallucination.ckpt-24023'    # modality hallucination
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul1_epochs50_nojoint_20180423-103045/multimodal_lambdamul1_epochs50_nojoint.ckpt-24000'    # mul_epochs=50
#model_path='/mnt/work/honda_100h/results/multimodal_base800_20180424-113707/multimodal_base800.ckpt-24008'    # base_model_800
#model_path='/mnt/work/honda_100h/results/multimodal_lambdamul1_epochs150_nopos_20180424-120717/multimodal_lambdamul1_epochs150_nopos.ckpt-24017'    # multimodal, no postive rows constraint
