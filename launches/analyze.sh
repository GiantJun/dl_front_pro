#!/usr/bin/env bash
set -e
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/src"

# --epoch=210 \
# --batch_size=16 \
# image_size 不能太小否则不能卷积
python "${SRC_DIR}"/main.py analyze\
  --cuda=3 \
  --model_type='res101' \
  --input_base_dir=./splited_dataset_noT/test \
  --model_load_dir=/home/yujun/Altrasound_pro/output/res50_w1/model \
  --grad_cam_dir=/home/yujun/Altrasound_pro/output/res50_w1/grad_cam_image \
  --show_log \