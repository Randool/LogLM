#!/bin/bash
# Author: Da Ma
# Date: 2021.12.31

set -e

DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
RAW_CORPORA_PATH="${DATA_PATH}/15.medical_plm_corpus"
TRAIN_NORMALIZED="${DATA_PATH}/normalized/train"
DEV_NORMALIZED="${DATA_PATH}/normalized/dev"
RATE=${RATE:-1}

set -x

PYTHONPATH=.. python normalize.py \
    --type              ${TYPE:-MLM} \
    --input_path        ${RAW_CORPORA_PATH} \
    --output_path       ${TRAIN_NORMALIZED} \
    --dev_output_path   ${DEV_NORMALIZED} \
    --num_doc_pre_file  ${num_doc_pre_file:-100000} \
    --num_dev_doc       ${num_dev_doc:-8000} \
    --rate              ${RATE}
