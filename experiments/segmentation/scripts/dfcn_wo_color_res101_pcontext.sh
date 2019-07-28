#!/usr/bin/env bash
#train
python train.py --dataset pcontext \
    --model fcn --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname fcn_wo_color_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model fcn --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/fcn/fcn_wo_color_res101_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model fcn --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/fcn/fcn_wo_color_res101_pcontext/model_best.pth.tar --split val --mode testval --ms