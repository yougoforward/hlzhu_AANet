#!/usr/bin/env bash
# python train.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 1024 --crop-size 768 --lr 0.01 --batch-size 8 --epochs 240 \
#     --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/checkpoint.pth.tar --ft \
#     --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval
# # train
# python train.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 1024 --crop-size 768 --lr 0.01 --batch-size 8 --epochs 240 \
#     --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval --train-split trainval

python train.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 1024 --crop-size 768 --lr 0.001 --batch-size 8 --epochs 5 \ 
    --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval --train-split trainval

python train.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 1024 --crop-size 768 --lr 0.01 --batch-size 8 --epochs 240 \
    --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --ft \
    --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval --train-split trainval
# finetune
python train_city.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 1024 --crop-size 768 --lr 0.0001 --epochs 240 --lr-scheduler step \
    --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/checkpoint.pth.tar --ft \
    --backbone resnet101 --checkname new_psp3_res101_cityscapes_trainval_finetune --no-val --train-split trainval
#test [single-scale]
# python test.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/checkpoint.pth.tar --split val --mode testval


#test [multi-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --multi-grid --base-size 2048 --crop-size 1024 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split test --mode test --ms

# #test [multi-scale]
# python test_whole_gpu.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 2048 --crop-size 1024 \
#     --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes_trainval/model_best.pth.tar --split test --mode test --ms