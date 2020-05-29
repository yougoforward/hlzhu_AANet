# !/usr/bin/env bash
# train
python train.py --dataset cocostuff \
    --model gsnet5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname gsnet5_res101_cocostuff --lr 0.001

#test [single-scale]
python test.py --dataset cocostuff \
    --model gsnet5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/cocostuff/gsnet5/gsnet5_res101_cocostuff/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset cocostuff \
    --model gsnet5 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/cocostuff/gsnet5/gsnet5_res101_cocostuff/model_best.pth.tar --split val --mode testval --ms