#!/usr/bin/env bash
#train
python train_lovasz.py --dataset pcontext \
    --model new_psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname new_psp6_lovasz_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model new_psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/new_psp6/new_psp6_lovasz_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model new_psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/new_psp6/new_psp6_lovasz_res101_pcontext/model_best.pth.tar --split val --mode testval --ms