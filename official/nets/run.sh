#!/usr/bin/env bash
PYTHONPATH=../../ python cifar10_main.py --data_dir=data/cifar10_data/ --num_gpus=4 --data_format="channels_last" --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type='STANDARD' >> cifar10_gpu_3_log_true_spatial_param-2018-08-16-13-39.log 2>&1 &
wait
PYTHONPATH=../../ python cifar10_main.py  --data_dir=data/cifar10_data/ --num_gpus=4 --data_format="channels_last" --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type='SPECTRAL_PARAM' >> cifar10_gpu_3_log_true_spectral_param-2018-08-16-13-39.log 2>&1 &
wait
