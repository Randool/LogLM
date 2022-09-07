#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --output=./preprocess-%A.log
#SBATCH --error=./preprocess-%A.log

set -e

TYPE='AR'
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"

NORMALIZED_TRAIN="${DATA_PATH}/normalized/train"
NORMALIZED_DEV="${DATA_PATH}/normalized/dev"

PROCESSED_TRAIN="${DATA_PATH}/processed/train"
PROCESSED_DEV="${DATA_PATH}/processed/dev"

MAX_PRED_PER_SEQ=40
MAX_SEQ_LEN=128
EPOCH=10
SEED=42

MERGE='true' # true | false
SHUFFLE='false' # true | false
RELEASE_CACHE='true' # true | false

VOCAB_PATH="${HOME}/Projects/LogLM/preprocess_data/tokenizer/vocab.txt"


PYTHONPATH=.. \
python preprocess.py \
    --type              ${TYPE} \
    --maxPredPerSeq     ${MAX_PRED_PER_SEQ} \
    --dupEpochNum       ${EPOCH} \
    --verboseStep       5000 \
    --vocabPath         ${VOCAB_PATH} \
    --maxSeqLen         ${MAX_SEQ_LEN} \
    --WWM \
    --input_path        ${NORMALIZED_TRAIN} \
    --output_path       ${PROCESSED_TRAIN} \
    --processNum        30 \
    --maskRatio         0.15 \
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
    --input_path         ${NORMALIZED_DEV} \
    --output_path        ${PROCESSED_DEV} \
    --processNum        30 \
    --maskRatio         0.15 \
    --logPath           "${DATA_PATH}/logs" \
    --seed              ${SEED}

cat ${PROCESSED_DEV}/*/* > "${PROCESSED_DEV}/dev.data"

if [[ ${RELEASE_CACHE} == 'true' ]]; then
    rm -rf ${PROCESSED_DEV}/epoch*
fi


# # Shuffle lines to prevent build random index of lines
# if [[ ${MERGE} == 'true' && ${SHUFFLE} == 'true' ]]; then
#     for i in `seq 0 $((${EPOCH} - 1))`; do
#         SHUF_SEED=$((${SEED} + ${i}))
#         raw_file="${PROCESSED_TRAIN}/epoch-${i}.pretrain.data"

#         echo "Shuffle [${SHUF_SEED}]: ${raw_file}"
#         shuf --random-source=<(yes ${SHUF_SEED}) "${raw_file}" > "${raw_file}.shuf"

#         if [[ ${RELEASE_CACHE} == 'true' ]]; then
#             rm ${raw_file}
#         fi
#     done
# fi

# Build index
python ../tools/create_dataset_index.py \
    --inputs "${PROCESSED_TRAIN}"'/*.pretrain.data'

python ../tools/create_dataset_index.py \
    --inputs "${PROCESSED_DEV}"'/dev.data'
