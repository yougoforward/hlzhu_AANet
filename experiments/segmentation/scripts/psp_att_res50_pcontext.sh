# !/usr/bin/env bash
# train
python train.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname psp_att_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/psp_att/psp_att_res50_pcontext/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model psp_att --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/psp_att/psp_att_res50_pcontext/checkpoint.pth.tar --split val --mode testval --ms