#!/usr/bin/env bash

for batch_norm_state in 'INACTIVE' 'ACTIVE'; do
    for conv_type in 'SPECTRAL_PARAM' 'STANDARD'; do
        rm -rf ../resnet/data/cifar10_model
        rm -rf data/cifar10_model
        TIMESTAMP=$(date +date-%F-time-%H-%M-%S-%N)
        echo "conv_type="${conv_type}
        echo "batch_norm_state="${batch_norm_state}
        echo "timestamp="${TIMESTAMP}
        PYTHONPATH=../../ python cifar10_main.py --model_type=RES_NET --optimizer_type=ADAM --data_dir=data/cifar10_data/ --model_dir=data/cifar10_model --num_gpus=4 --data_format="channels_last" --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type=${conv_type} --batch_norm_state=${batch_norm_state} >> cifar10_${HOSTNAME}_${conv_type}_${batch_norm_state}-${TIMESTAMP}.log 2>&1 &
        # echo "--model_type=RES_NET --optimizer_type=ADAM --data_dir=data/cifar10_data/ --model_dir=data/cifar10_model --num_gpus=4 --data_format="channels_last" --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type=${conv_type} --batch_norm_state=${batch_norm_state} >> cifar10_${HOSTNAME}_${conv_type}_${batch_norm_state}-${TIMESTAMP}.log 2>&1 &"
        wait
    done;
done;

# PYTHONPATH=../../ python cifar10_main.py --model_type=RES_NET --optimizer_type=ADAM --data_dir=data/cifar10_data/ --model_dir=data/cifar10_model/ --num_gpus=4 --data_format=channels_last --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type=SPECTRAL_PARAM --batch_norm_state=INACTIVE >> cifar10_athena_SPECTRAL_PARAM_INACTIVE-date-2018-08-17-time-13-21-22-599144623.log 2>&1 &
# PYTHONPATH=../../ python cifar10_main.py --model_type=RES_NET --optimizer_type=ADAM --data_dir=data/cifar10_data/ --model_dir=data/cifar10_model/ --num_gpus=4 --data_format=channels_last --train_epochs=300 -ebe=1 --resnet_size=32 --conv_type=STANDARD --batch_norm_state=INACTIVE >> cifar10_gpu2_STANDARD_INACTIVE-date-2018-08-17-time-13-21-22-599144623.log 2>&1 &