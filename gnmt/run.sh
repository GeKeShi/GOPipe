#!/bin/bash

set -e

DATASET_DIR='data/wmt16_de_en'
TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64

SEED=${1:-"1"}
TARGET=${2:-"24.00"}

# run training
python3 train.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --train-batch-size ${TRAIN_BATCH_SIZE} \
  --test-batch-size ${TEST_BATCH_SIZE}
