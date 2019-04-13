#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model dict_aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50 --checkname dict_aanet_seloss_jpu_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model dict_aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50 --resume runs/pcontext/dict_aanet/dict_aanet_seloss_jpu_res50_pcontext/model_best.pth.tar \
    --split val --mode testval