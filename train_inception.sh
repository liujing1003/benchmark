#!/usr/bin/env bash

python main.py \
    --dataset webvision \
    --train_image_size 299 \
    --test_image_size 342 \
    --test_crop_image_size 299 \
    --model Inception \
    --depth 18 \
    --lr 0.01 \
    --batchsize 16 \
    --print-freq 10 \
    --expname Inception \
    --resume /media/ouc/30bd7817-d3a1-4e83-b7d9-5c0e373ae434/LiuJing/WebVision/webvision_checkpoints/Inception/checkpoint.pth.tar \
    --gpu_ids 2
