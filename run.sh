#!/usr/bin/env bash

PROG=$1

BATCHSIZE=256

ARGS="--seed=1 \
      --gpus=1 \
      --max_epochs=40 \
      --num_workers=4 \
      --train_batch_size=$BATCHSIZE \
      --val_batch_size=$BATCHSIZE \
      --learning_rate=1e-3 \
      --weight_decay=1e-2"

python3 $PROG $ARGS
