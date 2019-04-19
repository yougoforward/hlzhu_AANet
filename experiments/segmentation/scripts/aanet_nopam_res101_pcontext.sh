#!/usr/bin/env bash

#train
python train.py --dataset cocostuff \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_nopam_res101_cocostuff

#test [single-scale]
python test.py --dataset cocostuff \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet_nopam/aanet_nopam_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset cocostuff \
    --model aanet_nopam --aux --dilated --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/cocostuff/aanet_nopam/aanet_nopam_res101_cocostuff/model_best.pth.tar \
    --split val --mode testval --ms