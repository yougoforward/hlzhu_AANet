#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname psp6_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/psp6/psp6_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psp6 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/psp6/psp6_res50_pcontext/model_best.pth.tar --split val --mode testval --ms