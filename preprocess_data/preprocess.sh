#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --output=./preprocess-%A.log
#SBATCH --error=./preprocess-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

TYPE='MLM'
MERGE='true' # true | false
SHUFFLE='true' # true | false
RELEASE_CACHE='true' # true | false
MAX_PRED_PER_SEQ=40
MAX_SEQ_LEN=128
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
INPUT_PATH="${DATA_PATH}/normalized"
OUTPUT_PATH="${DATA_PATH}/processed/train"
VOCAB_PATH="${HOME}/Projects/LogLM/exp/tokenizer/vocab.txt"
EPOCH=10
SHUF_SEED=42

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
    --processNum    30 \
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

# Shuffle lines to prevent build random index of lines
if [[ ${SHUFFLE} == 'true' ]]; then
    for i in `seq 0 $((${EPOCH} - 1))`; do
        SEED=$((${SHUF_SEED} + ${i}))
        raw_file="${OUTPUT_PATH}/epoch-${i}.pretrain.data"
        echo "Shuffle [${SEED}]: ${raw_file}"
        shuf --random-source=<(yes ${SEED}) "${raw_file}" > "${raw_file}.shuf"
    done
fi
