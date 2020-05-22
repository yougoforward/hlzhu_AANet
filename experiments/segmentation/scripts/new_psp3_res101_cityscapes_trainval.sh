#!/usr/bin/env bash
# train
python train.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 1024 --crop-size 769 --lr 0.001 --batch-size 8 --epochs 240 \
    --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval --train-split trainval

#test [single-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 769 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split test --mode test --ms --save-folder cityscapes_best