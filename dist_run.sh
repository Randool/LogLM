#!/bin/bash

# Need:
# - MASTER_HOST
# - MASTER_PORT
# - MODEL_SIZE= tiny | base | large
# - nproc_per_node
# - PER_DEVICE_TRAIN_BATCH_SIZE
# - PER_DEVICE_EVAL_BATCH_SIZE
# - GRADIENT_ACCUMULATION_STEPS
# - EVAL_ACCUMULATION_STEPS

legal=true

for slot in MASTER_HOST MASTER_PORT MODEL_SIZE nproc_per_node PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS EVAL_ACCUMULATION_STEPS; do
  if [ -z ${slot} ]; then
    echo "No ${slot}"
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
CORPUS_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
TRAIN_DATA_PATH="${CORPUS_PATH}/processed/train"
EVAL_DATA_PATH="${CORPUS_PATH}/processed/dev"
VOCAB_PATH="${CORPUS_PATH}/tokenizer/vocab.txt"

# ==== training hyperparameters ==== #
LEARNING_RATE=1e-4
#PER_DEVICE_TRAIN_BATCH_SIZE=16
#PER_DEVICE_EVAL_BATCH_SIZE=16
#GRADIENT_ACCUMULATION_STEPS=2
#EVAL_ACCUMULATION_STEPS=2
WEIGHT_DECAY=0.01
MAX_STEPS=1000000
NUM_TRAIN_EPOCHS=10
LR_SCHEDULER_TYPE="linear"
WARMUP_STEPS=10000
LOGGING_STEPS=1000
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=30
EVALUATION_STRATEGY="steps"
EVAL_STEPS=50000
DATALOADER_NUM_WORKERS=6
LABEL_SMOOTHING_FACTOR=0
DDP_FIND_UNUSED_PARAMETERS=false
NUM_EXAMPLES_FOR_EPOCH="38944777 38944777 38944777 38944777 38944777 38944777 17901367"

# ==== train path ==== #
OUTPUT_DIR="${CORPUS_PATH}/ckpts/${MODEL_SIZE}"
LOG_PATH="${OUTPUT_DIR}" # None means output to the stdout
TENSORBOARD_LOG_PATH="${OUTPUT_DIR}/tensorboard"

if [ -z ${RESUME_CHECKPOINT_PATH} ]; then
  RESUME_CHECKPOINT_PATH=None
fi

# *************************************************** #
if [[ ${MODEL_SIZE} == 'tiny' ]]; then
    HIDDEN_SIZE=256
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=8
    INTERMEDIATE_SIZE=1024
    NON_CASUAL_LAYERS=2
    CASUAL_LAYERS=2
elif [[ ${MODEL_SIZE} == 'base' ]]; then
    HIDDEN_SIZE=768
    NUM_HIDDEN_LAYERS=12
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=3072
    NON_CASUAL_LAYERS=3
    CASUAL_LAYERS=3
else  # large
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=24
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=4096
    NON_CASUAL_LAYERS=6
    CASUAL_LAYERS=6
fi

OMP_NUM_THREADS=1 \
python -um torch.distributed.launch \
    --nnodes          ${SLURM_NNODES} \
    --node_rank       ${SLURM_NODEID} \
    --nproc_per_node  ${nproc_per_node} \
    --master_addr     ${MASTER_HOST} \
    --master_port     ${MASTER_PORT} \
main.py \
    --seed                          ${SEED} \
    --trainDataPath                 ${TRAIN_DATA_PATH} \
    --evalDataPath                  ${EVAL_DATA_PATH} \
    --vocabPath                     ${VOCAB_PATH} \
    --logPath                       ${LOG_PATH} \
    --resume_checkpoint             ${RESUME_CHECKPOINT_PATH} \
    --hidden_size                   ${HIDDEN_SIZE} \
    --num_hidden_layers             ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads           ${NUM_ATTENTION_HEADS} \
    --intermediate_size             ${INTERMEDIATE_SIZE} \
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
    --ddp_find_unused_parameters    ${DDP_FIND_UNUSED_PARAMETERS} \
    --load_best_model_at_end        true \
    --metric_for_best_model         mlm_acc \
    --greater_is_better             true \
    \
    --num_examples_for_epoch        ${NUM_EXAMPLES_FOR_EPOCH}
