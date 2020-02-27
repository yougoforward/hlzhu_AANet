#!/usr/bin/env bash
# #test [single-scale]
# python test_whole_gpu.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval

# #test [multi-scale]
# python test_whole_gpu.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval --ms

#test [single-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval --ms