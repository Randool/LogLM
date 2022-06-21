#!/bin/bash
# Author: Da Ma
# Date: 2022.01.05

set -e

DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
INPUT_DIR="${DATA_PATH}/15.medical_plm_corpus"

INPUT_FILE_LIST=''
for file in $(ls ${INPUT_DIR}); do
    INPUT_FILE_LIST="${INPUT_FILE_LIST} ${INPUT_DIR}/${file}"
done

SAVE_PATH="${DATA_PATH}/tokenizer"

python train.py \
    --inputFileList ${INPUT_FILE_LIST} \
    --savePath ${SAVE_PATH} \
    --specialTokens '[PAD]' '[CLS]' '[UNK]' '[SEP]' '[MASK]'
