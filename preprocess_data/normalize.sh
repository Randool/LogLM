#!/bin/bash
# Author: Da Ma
# Date: 2021.12.31

set -e

TYPE='MLM'
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
RAW_CORPORA_PATH="${DATA_PATH}/15.medical_plm_corpus"
TRAIN_NORMALIZED="${DATA_PATH}/normalized"
DEV_NORMALIZED="${DATA_PATH}/normalized_dev"
RATE=1

PYTHONPATH=.. python normalize.py \
    --type            ${TYPE} \
    --inputPath       ${RAW_CORPORA_PATH} \
    --outputPath      ${TRAIN_NORMALIZED} \
    --dev_output_path ${DEV_NORMALIZED} \
    --docNumPerFile   100000 \
    --num_dev_doc     10000 \
    --rate ${RATE}
