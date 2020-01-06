# #!/usr/bin/env bash
# train
# python train_city_weight.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 1024 --crop-size 768 --lr 0.01 --epochs 240 \
#     --backbone resnet101 --checkname new_psp3_res101_cityscapes

# python train_city.py --dataset cityscapes \
#     --model new_psp3 --aux --dilated --base-size 1024 --crop-size 768 --lr 0.001 --epochs 240 \
#     --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/checkpoint.pth.tar --ft \
#     --backbone resnet101 --checkname new_psp3_res101_cityscapes_finetune
#test [single-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --base-size 2048 --crop-size 768 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset cityscapes \
    --model new_psp3 --aux --dilated --base-size 2048 --crop-size 1024 \
    --backbone resnet101 --resume runs/cityscapes/new_psp3/new_psp3_res101_cityscapes/checkpoint.pth.tar --split val --mode testval --ms