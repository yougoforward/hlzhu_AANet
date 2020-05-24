# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model gsnet8 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname gsnet8_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model gsnet8 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet8/gsnet8_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model gsnet8 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet8/gsnet8_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms