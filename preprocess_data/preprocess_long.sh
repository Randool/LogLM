#!/bin/bash
#SBATCH --job-name=preprocess_long
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=50
#SBATCH --mem=30G
#SBATCH --output=./preprocess_long-%A.log
#SBATCH --error=./preprocess_long-%A.log

set -e

TYPE='t5'  # mlm | bart | t5
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"

NORMALIZED_TRAIN="${DATA_PATH}/normalized/train"
NORMALIZED_DEV="${DATA_PATH}/normalized/dev"

PROCESSED_TRAIN="${DATA_PATH}/processed/trainL"
PROCESSED_DEV="${DATA_PATH}/processed/devL"

MAX_SEQ_LEN=512
MAX_PRED_PER_SEQ=76
MAX_SEQ_LEN2=2048
MAX_PRED_PER_SEQ2=256
maskRatio=0.15
EPOCH=4
SEED=42

MERGE='true' # true | false
RELEASE_CACHE='true' # true | false

VOCAB_PATH="${HOME}/Projects/LogLM/preprocess_data/tokenizer/vocab.txt"

set -x

PYTHONPATH=.. \
python preprocess.py \
    --type              ${TYPE} \
    \
    --maxSeqLen         ${MAX_SEQ_LEN} \
    --maxPredPerSeq     ${MAX_PRED_PER_SEQ} \
    \
    --maxSeqLen2        ${MAX_SEQ_LEN2} \
    --maxPredPerSeq2    ${MAX_PRED_PER_SEQ2} \
    \
    --dupEpochNum       ${EPOCH} \
    --verboseStep       5000 \
    --vocabPath         ${VOCAB_PATH} \
    --WWM \
    --input_path        ${NORMALIZED_TRAIN} \
    --output_path       ${PROCESSED_TRAIN} \
    --processNum        30 \
    --maskRatio         ${maskRatio} \
    --logPath           "${DATA_PATH}/logs" \
    --seed              ${SEED}


# merge data
if [[ ${MERGE} == 'true' ]]; then
    for epoch in $(ls ${PROCESSED_TRAIN}); do
        EPOCH_PATH="${PROCESSED_TRAIN}/${epoch}"

        if [[ -d ${EPOCH_PATH} ]]; then
            echo "merging data in: ${EPOCH_PATH}"
            cat ${EPOCH_PATH}/* > "${EPOCH_PATH}.pretrain.data"
            if [[ ${RELEASE_CACHE} == 'true' ]]; then
                rm -rf ${EPOCH_PATH}
            fi
        fi
    done
fi

# Dev
PYTHONPATH=.. \
python preprocess.py \
    --type              ${TYPE} \
    --maxPredPerSeq     ${MAX_PRED_PER_SEQ} \
    --dupEpochNum       1 \
    --verboseStep       5000 \
    --vocabPath         ${VOCAB_PATH} \
    --maxSeqLen         ${MAX_SEQ_LEN} \
    --WWM \
    --input_path        ${NORMALIZED_DEV} \
    --output_path       ${PROCESSED_DEV} \
    --processNum        30 \
    --maskRatio         ${maskRatio} \
    --logPath           "${DATA_PATH}/logs" \
    --seed              ${SEED}

cat ${PROCESSED_DEV}/*/* > "${PROCESSED_DEV}/dev.data"

if [[ ${RELEASE_CACHE} == 'true' ]]; then
    rm -rf ${PROCESSED_DEV}/epoch*
fi

# Build index
python ../tools/create_dataset_index.py \
    --inputs "${PROCESSED_TRAIN}"'/*.pretrain.data'

python ../tools/create_dataset_index.py \
    --inputs "${PROCESSED_DEV}"'/dev.data'
