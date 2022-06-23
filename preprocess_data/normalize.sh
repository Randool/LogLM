#!/bin/bash
# Author: Da Ma
# Date: 2021.12.31

set -e

TYPE='MLM'
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
INPUT_PATH="${DATA_PATH}/15.medical_plm_corpus"
OUTPUT_PATH="${DATA_PATH}/normalized"
RATE=1

PYTHONPATH=.. python normalize.py \
    --type ${TYPE} \
    --inputPath ${INPUT_PATH} \
    --outputPath ${OUTPUT_PATH} \
    --rate ${RATE}
