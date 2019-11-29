#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model new_psp5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname new_psp5_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model new_psp5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/new_psp5/new_psp5_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model new_psp5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/new_psp5/new_psp5_res50_pcontext/model_best.pth.tar --split val --mode testval --ms