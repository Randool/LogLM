#!/bin/bash

# Need:
# - MASTER_HOST
# - MASTER_PORT
# - nproc_per_node
# - PER_DEVICE_TRAIN_BATCH_SIZE
# - PER_DEVICE_EVAL_BATCH_SIZE
# - GRADIENT_ACCUMULATION_STEPS
# - EVAL_ACCUMULATION_STEPS
# - MODEL_SIZE= tiny | base | large
# - HIDDEN_SIZE
# - NUM_HIDDEN_LAYERS
# - NUM_ATTENTION_HEADS
# - INTERMEDIATE_SIZE
# - NON_CASUAL_LAYERS
# - CASUAL_LAYERS
# - SHARED_NON_CASUAL_DECODER

legal=true

for slot in MASTER_HOST MASTER_PORT nproc_per_node PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS EVAL_ACCUMULATION_STEPS; do
  if [ -z "${slot}" ]; then
    echo "Require ${slot}"
    legal=false
  fi
done

for slot in MODEL_SIZE HIDDEN_SIZE NUM_HIDDEN_LAYERS NUM_ATTENTION_HEADS INTERMEDIATE_SIZE NON_CASUAL_LAYERS CASUAL_LAYERS SHARED_NON_CASUAL_DECODER; do
  if [ -z "${slot}" ]; then
    echo "Require ${slot}"
    legal=false
  fi
done

if [[ ${legal} == false ]]; then
  exit -1
fi

echo "<node_rank=${SLURM_NODEID}, nnodes=${SLURM_NNODES}> | MASTER_HOST=${MASTER_HOST}, \
MASTER_PORT=${MASTER_PORT} | MODEL_SIZE=${MODEL_SIZE}, nproc_per_node=${nproc_per_node}"

set -e

# ******************* CONFIGURATION ******************* #
# ==== data path ==== #
CORPUS_PATH=${CORPUS_PATH:-"${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"}
TRAIN_DATA_PATH="${CORPUS_PATH}/processed/train"
EVAL_DATA_PATH="${CORPUS_PATH}/processed/dev"
VOCAB_PATH=${VOCAB_PATH:-"${CORPUS_PATH}/tokenizer/vocab.txt"}

LOG_BASE=${LOG_BASE:-2}

# ==== training hyper-parameters ==== #
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
MAX_STEPS=${MAX_STEPS:-1000000}
NUM_TRAIN_EPOCHS=10
LR_SCHEDULER_TYPE="linear"
WARMUP_STEPS=10000
LOGGING_STEPS=1000
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=20
EVALUATION_STRATEGY="steps"
EVAL_STEPS=50000
DATALOADER_NUM_WORKERS=6
LABEL_SMOOTHING_FACTOR=0

if [ -z "${NUM_EXAMPLES_FOR_EPOCH}" ]; then
  NUM_EXAMPLES_FOR_EPOCH="38944777 38944777 38944777 17901367"
fi

SHARED_NON_CASUAL_DECODER=${SHARED_NON_CASUAL_DECODER:-false}


# ==== train path ==== #
OUTPUT_DIR=${CORPUS_PATH}/ckpts/${MODEL_SIZE}

# +非自回归层数
if ${SHARED_NON_CASUAL_DECODER}; then
  OUTPUT_DIR=${OUTPUT_DIR}_shared_ndec
elif [ ${NON_CASUAL_LAYERS} -gt 0 ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_ndec${NON_CASUAL_LAYERS}
fi

# +自回归层数
if [ ${CASUAL_LAYERS} -gt 0 ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_dec${CASUAL_LAYERS}
fi

# +log底数
OUTPUT_DIR=${OUTPUT_DIR}_log${LOG_BASE}

LOG_PATH="${OUTPUT_DIR}" # None means output to the stdout
TENSORBOARD_LOG_PATH="${OUTPUT_DIR}/tensorboard"
RESUME_CHECKPOINT_PATH=${RESUME_CHECKPOINT_PATH:-None}

echo $OUTPUT_DIR

OMP_NUM_THREADS=1 \
python -um torch.distributed.launch \
    --nnodes          ${SLURM_NNODES} \
    --node_rank       ${SLURM_NODEID} \
    --nproc_per_node  ${nproc_per_node} \
    --master_addr     ${MASTER_HOST} \
    --master_port     ${MASTER_PORT} \
main.py \
    --seed                          ${SEED} \
    --fp16                          false \
    --ignore_data_skip              false \
    \
    --trainDataPath                 ${TRAIN_DATA_PATH} \
    --evalDataPath                  ${EVAL_DATA_PATH} \
    --vocabPath                     ${VOCAB_PATH} \
    --logPath                       ${LOG_PATH} \
    --log_base                      ${LOG_BASE} \
    \
    --resume_checkpoint             ${RESUME_CHECKPOINT_PATH} \
    --hidden_size                   ${HIDDEN_SIZE} \
    --num_hidden_layers             ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads           ${NUM_ATTENTION_HEADS} \
    --intermediate_size             ${INTERMEDIATE_SIZE} \
    --shared_non_casual_decoder     ${SHARED_NON_CASUAL_DECODER} \
    --num_non_casual_decoder_layers ${NON_CASUAL_LAYERS} \
    --num_casual_decoder_layers     ${CASUAL_LAYERS} \
    \
    --output_dir                    ${OUTPUT_DIR} \
    --learning_rate                 ${LEARNING_RATE} \
    --evaluation_strategy           ${EVALUATION_STRATEGY} \
    --per_device_train_batch_size   ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size    ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps   ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_accumulation_steps       ${EVAL_ACCUMULATION_STEPS} \
    --weight_decay                  ${WEIGHT_DECAY} \
    --max_steps                     ${MAX_STEPS} \
    --num_train_epochs              ${NUM_TRAIN_EPOCHS} \
    --lr_scheduler_type             ${LR_SCHEDULER_TYPE} \
    --warmup_steps                  ${WARMUP_STEPS} \
    --logging_dir                   ${TENSORBOARD_LOG_PATH} \
    --logging_first_step            true \
    --logging_steps                 ${LOGGING_STEPS} \
    \
    --save_steps                    ${SAVE_STEPS} \
    --save_total_limit              ${SAVE_TOTAL_LIMIT} \
    --eval_steps                    ${EVAL_STEPS} \
    --dataloader_num_workers        ${DATALOADER_NUM_WORKERS} \
    --label_smoothing_factor        ${LABEL_SMOOTHING_FACTOR} \
    --ddp_find_unused_parameters    true \
    --load_best_model_at_end        true \
    --metric_for_best_model         mlm_acc \
    --greater_is_better             true \
    \
    --num_examples_for_epoch        ${NUM_EXAMPLES_FOR_EPOCH}
