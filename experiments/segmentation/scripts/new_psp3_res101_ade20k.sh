#!/usr/bin/env bash
#train
python train.py --dataset ade20k \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname new_psp3_res101_ade20k --no-val

#test [single-scale]
python test.py --dataset ade20k \
    --model new_psp3 --aux --dilated --base-size 576 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/new_psp3/new_psp3_res101_ade20k/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset ade20k \
    --model new_psp3 --aux --dilated --base-size 576 --crop-size 576 \
    --backbone resnet101 --resume runs/ade20k/new_psp3/new_psp3_res101_ade20k/checkpoint.pth.tar --split val --mode testval --ms