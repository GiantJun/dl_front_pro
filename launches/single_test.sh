#!/usr/bin/env bash
set -e
export PYTHONPATH=`pwd`:$PYTHONPATH

WORK_DIR=$(pwd)
SRC_DIR="${WORK_DIR}/scripts"

# emsemble_test.py or single_test.py
python "${SRC_DIR}"/single_test.py test\
  --cuda=2 \
  --dataset=Altrasound \
  --num_workers=4 \
  --batch_size=6 \
  --model_load_dir=output/tnt_s_epoch200/model \
  --show_log \