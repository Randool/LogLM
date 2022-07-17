#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --output=./pipeline-%A.log
#SBATCH --error=./pipeline-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

phase=1

TYPE='MLM'
DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
RAW_CORPORA_PATH="${DATA_PATH}/15.medical_plm_corpus"

NORMALIZED_TRAIN="${DATA_PATH}/normalized"
NORMALIZED_DEV="${DATA_PATH}/normalized_dev"

PROCESSED_TRAIN="${DATA_PATH}/processed/train"
PROCESSED_DEV="${DATA_PATH}/processed/dev"

MAX_PRED_PER_SEQ=40
MAX_SEQ_LEN=128
EPOCH=7
SEED=42
NUM_WORKERS=30

MERGE='true' # true | false
SHUFFLE='true' # true | false
RELEASE_CACHE='true' # true | false

VOCAB_PATH="./tokenizer/vocab.txt"

# Phase 1: Normalized and split train & dev
if [[ ${phase} -le 1 ]]; then
    PYTHONPATH=.. \
    python normalize.py \
        --type              ${TYPE} \
        --inputPath         ${RAW_CORPORA_PATH} \
        --outputPath        ${NORMALIZED_TRAIN} \
        --dev_output_path   ${NORMALIZED_DEV} \
        --docNumPerFile     100000 \
        --num_dev_doc       10000 \
        --rate              1 \
        --seed              ${SEED}
fi

# Phase 2: Pre-process train & dev
if [[ ${phase} -le 2 ]]; then
    PYTHONPATH=.. \
    python preprocess.py \
        --type              ${TYPE} \
        --maxPredPerSeq     ${MAX_PRED_PER_SEQ} \
        --dupEpochNum       ${EPOCH} \
        --verboseStep       5000 \
        --vocabPath         ${VOCAB_PATH} \
        --maxSeqLen         ${MAX_SEQ_LEN} \
        --WWM \
        --inputPath         ${NORMALIZED_TRAIN} \
        --outputPath        ${PROCESSED_TRAIN} \
        --processNum        ${NUM_WORKERS} \
        --maskRatio         0.15 \
        --logPath           "${DATA_PATH}/logs" \
        --seed              ${SEED}

    PYTHONPATH=.. \
    python preprocess.py \
        --type              ${TYPE} \
        --maxPredPerSeq     ${MAX_PRED_PER_SEQ} \
        --dupEpochNum       1 \
        --verboseStep       5000 \
        --vocabPath         ${VOCAB_PATH} \
        --maxSeqLen         ${MAX_SEQ_LEN} \
        --WWM \
        --inputPath         ${NORMALIZED_DEV} \
        --outputPath        ${PROCESSED_DEV} \
        --processNum        ${NUM_WORKERS} \
        --maskRatio         0.15 \
        --logPath           "${DATA_PATH}/logs" \
        --seed              ${SEED}
fi

# Phase 3: Merge train data
if [[ ${phase} -le 3 ]]; then
    # Train
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
    cat ${PROCESSED_DEV}/*/* > "${PROCESSED_DEV}/dev.data"

    if [[ ${RELEASE_CACHE} == 'true' ]]; then
        rm -rf ${PROCESSED_DEV}/epoch*
    fi
fi

# Phase 4: Shuffle lines to prevent build random index of lines
if [[ ${phase} -le 3 ]]; then
    if [[ ${MERGE} == 'true' && ${SHUFFLE} == 'true' ]]; then
        for i in `seq 0 $((${EPOCH} - 1))`; do
            SHUF_SEED=$((${SEED} + ${i}))
            raw_file="${PROCESSED_TRAIN}/epoch-${i}.pretrain.data"
            echo "Shuffle [${SHUF_SEED}]: ${raw_file}"
            shuf --random-source=<(yes ${SHUF_SEED}) "${raw_file}" > "${raw_file}.shuf"

            if [[ ${RELEASE_CACHE} == 'true' ]]; then
                rm ${raw_file}
            fi
        done
    fi
fi
