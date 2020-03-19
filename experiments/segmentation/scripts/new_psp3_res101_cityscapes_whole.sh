#!/usr/bin/env bash
# #test [single-scale]
# python test_whole_gpu.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval

# #test [multi-scale]
# python test_whole_gpu.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval --ms

# #test [single-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval --ms


# #test [single-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval

# #test [single-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_finetune/model_best.pth.tar --split val --mode testval
    
# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split val --mode testval --ms

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_finetune/model_best.pth.tar --split val --mode testval --ms

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split test --mode test --ms --save-folder cityscapes_best

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --split test --mode test --ms --save-folder cityscapes_last


# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --split test --mode testval --ms --save-folder cityscapes_test

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --split val --mode testval
# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --split trainval --mode testval

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split val --mode testval

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split trainval --mode testval

# #test [multi-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/model_best.pth.tar --split test --mode test --ms --save-folder cityscapes_traintest

#test [multi-scale]
python test_whole_cpu.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split test --mode test --save-folder cityscapes_whole