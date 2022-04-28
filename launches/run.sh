#!/usr/bin/env bash
set -e
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/scripts"

python "${SRC_DIR}"/main.py train\
  --cuda=0 \
  --dataset=DLDataset \
  --model_type=swin_transformer \
  --save_dir_name=swin_transformer_batch16 \
  --num_workers=8 \
  --kfold=5 \
  --epoch=150 \
  --lr=0.0001 \
  --lr_epoch=40 \
  --batch_size=16 \
  --optimizer=adam \
  --show_log \
  # --scheduler=coslr \