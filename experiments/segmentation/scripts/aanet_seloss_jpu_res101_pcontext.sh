#!/usr/bin/env bash

#train
python train.py --dataset pcontext \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --checkname aanet_seloss_jpu_res101_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet/aanet_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model aanet --aux --jpu --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101 --resume runs/pcontext/aanet/aanet_seloss_jpu_res101_pcontext/model_best.pth.tar \
    --split val --mode testval --ms



#fps
CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet --jpu --aux --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet --jpu --aux --se-loss \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model dict_aanet --jpu --aux --se-loss \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet_nopam --jpu --aux --se-loss \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet --dilated --aux --se-loss \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model dict_aanet --dilated --aux --se-loss \
    --backbone resnet101

CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model encnet --dilated --aux --se-loss \
    --backbone resnet101


CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet --jpu --aux --se-loss --base-size 608 --crop-size 576 \
    --backbone resnet50


CUDA_VISIBLE_DEVICES=7 python test_fps_params.py --dataset pcontext \
    --model aanet --jpu --aux --se-loss \
    --backbone resnet50