#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname deeplabv3_att_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/deeplabv3_att/deeplabv3_att_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model deeplabv3_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/deeplabv3_att/deeplabv3_att_res101_pcontext/model_best.pth.tar \
     --split val --mode testval --ms