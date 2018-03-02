#!/usr/bin/env bash

python main.py \
    --dataset webvision \
    --model VGG \
    --depth 16 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --batchsize 16 \
    --print-freq 10 \
    --expname vgg16 \
    --tensorboard \
    --resume /media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/webvision_checkpoints/vgg16/checkpoint.pth.tar \
    --gpu_ids 2
