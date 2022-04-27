#!/usr/bin/env bash
set -e
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/scripts"

python "${SRC_DIR}"/main_multi_branch.py train\
  --cuda=5 \
  --dataset=Altrasound \
  --model_type=res18 \
  --num_workers=4 \
  --epoch=150 \
  --lr=0.0001 \
  --lr_epoch=40 \
  --batch_size=8 \
  --optimizer=adam \
  --show_log \
  --model_load_dir=/home/21/yujun/altrasound_pro/output/res18_pretrain/model \
  # --select_list 0 1