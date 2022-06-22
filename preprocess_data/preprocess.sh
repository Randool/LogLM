#!/bin/bash
# Author: Da Ma
# Date: 2021.01.04

set -e

TYPE='MLM'
MERGE='true' # true | false
RELEASE_CACHE='true' # true | false
MAX_PRED_PER_SEQ=80
MAX_SEQ_LEN=128
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
INPUT_PATH="${DATA_PATH}/normalized"
OUTPUT_PATH="${DATA_PATH}/processed/train"
VOCAB_PATH="${HOME}/Projects/LogLM/exp/tokenizer/vocab.txt"
EPOCH=10

PYTHONPATH=".." python preprocess.py \
    --type          ${TYPE} \
    --maxPredPerSeq ${MAX_PRED_PER_SEQ} \
    --dupEpochNum   ${EPOCH} \
    --verboseStep   5000 \
    --vocabPath     ${VOCAB_PATH} \
    --maxSeqLen     ${MAX_SEQ_LEN} \
    --WWM \
    --inputPath     ${INPUT_PATH} \
    --outputPath    ${OUTPUT_PATH} \
    --processNum    20 \
    --maskRatio     0.15 \
    --logPath       "${DATA_PATH}/logs"

# merge data
if [[ ${MERGE} == 'true' ]]; then
    for epoch in $(ls ${OUTPUT_PATH}); do
        EPOCH_PATH="${OUTPUT_PATH}/${epoch}"

        if [[ -d ${EPOCH_PATH} ]]; then
            echo "merging data in: ${EPOCH_PATH}"
            cat ${EPOCH_PATH}/* > "${EPOCH_PATH}.pretrain.data"
            if [[ ${RELEASE_CACHE} == 'true' ]]; then
                rm -rf ${EPOCH_PATH}
            fi
        fi
    done
fi
