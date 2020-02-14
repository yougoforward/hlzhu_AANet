#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname new_psp3_nogrid_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/new_psp3/new_psp3_nogrid_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model new_psp3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/new_psp3/new_psp3_nogrid_res101_pcontext/model_best.pth.tar --split val --mode testval --ms