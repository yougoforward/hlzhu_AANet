#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pcontext \
    --model pgfnet --jpu --aux --se_loss \
    --backbone resnet50 --checkname pgfnet_res50_pcontext

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model pgfnet --jpu --aux --se_loss \
    --backbone resnet50 --resume runs/pcontext/pgfnet/pgfnet_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pcontext \
    --model pgfnet --jpu --aux --se_loss \
    --backbone resnet50 --resume runs/pcontext/pgfnet/pgfnet_res50_pcontext/model_best.pth.tar --split val --mode testval --ms