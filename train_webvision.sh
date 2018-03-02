#!/usr/bin/env bash

python main.py \
    --dataset webvision \
    --model AlexNet \
    --depth 152 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --batchsize 16 \
    --print-freq 10 \
    --expname AlexNet \
    --tensorboard \
    --resume /media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/webvision_checkpoints/AlexNet/checkpoint.pth.tar \
    --gpu_ids 3 
