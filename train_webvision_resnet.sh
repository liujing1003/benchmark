#!/usr/bin/env bash

python main.py \
    --dataset webvision \
    --model ResNet \
    --depth 152 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --batchsize 16 \
    --print-freq 10 \
    --expname ResNet-152 \
    --tensorboard \
    --gpu_ids 3 \
