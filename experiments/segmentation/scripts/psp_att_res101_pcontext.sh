# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --checkname psp_att_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/psp_att/psp_att_res101_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet101 --resume runs/pcontext/psp_att/psp_att_res101_pcontext/checkpoint.pth.tar --split val --mode testval --ms