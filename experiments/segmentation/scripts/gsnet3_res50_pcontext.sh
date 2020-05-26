# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model gsnet3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname gsnet3_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model gsnet3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet3/gsnet3_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model gsnet3 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/gsnet3/gsnet3_res50_pcontext/model_best.pth.tar --split val --mode testval --ms