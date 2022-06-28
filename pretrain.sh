#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --partition=a10
#SBATCH --cpus-per-task=4
#SBATCH --gres=1
#SBATCH --mem=16G
#SBATCH --output=./pretrain-%A.log
#SBATCH --error=./pretrain-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]
#SBATCH --array=0,1,2,3

set -e

# ******************* CONFIGURATION ******************* #
# ==== path ==== #
CORPUS_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
TRAIN_DATA_PATH="${CORPUS_PATH}/processed/train"
EVAL_DATA_PATH="${CORPUS_PATH}/processed/dev"
VOCAB_PATH="${CORPUS_PATH}/tokenizer/vocab.txt"

# ==== devices ==== #
NODE_NUM=1
NODE_RANK=0
PROC_NUM_PER_NODE=8
MASTER_IP="127.0.0.1"
MASTER_PORT=8888

# ==== training hyperparameters ==== #
MODEL_SIZE="base" # base | large
LEARNING_RATE=1e-3
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
EVAL_ACCUMULATION_STEPS=4
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
DATALOADER_NUM_WORKERS=4
LABEL_SMOOTHING_FACTOR=0.1
DDP_FIND_UNUSED_PARAMETERS=false
NUM_EXAMPLES_FOR_EPOCH="38944777 38944777 38944777 38944777 38944777 38944777 38944777 38944777 38944777 17901367"

# ******************* PREPARATION ******************* #
OUTPUT_DIR="${CORPUS_PATH}/ckpts/${MODEL_SIZE}"
LOG_PATH="${OUTPUT_DIR}/log" # None means output to the stdout
TENSORBOARD_LOG_PATH="${OUTPUT_DIR}/tensorboard"

RESUME_CHECKPOINT_PATH=None

# *************************************************** #

if [[ ${MODEL_SIZE} == 'tiny' ]]; then
    HIDDEN_SIZE=256
    NUM_HIDDEN_LAYERS=6
    NUM_ATTENTION_HEADS=8
    INTERMEDIATE_SIZE=1024
elif [[ ${MODEL_SIZE} == 'base' ]]; then
    HIDDEN_SIZE=768
    NUM_HIDDEN_LAYERS=12
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=3072
else
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=24
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=4096
fi

# ***************************************************** #
# NCCL_DEBUG=INFO  #** OPEN THE NCCL DEBUG **#
# NCCL_SOCKET_IFNAME="bond0" # ** SET THE INTERFACE OF NET ** #

python -um torch.distributed.launch \
    --nproc_per_node ${PROC_NUM_PER_NODE} \
    --nnodes ${NODE_NUM} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_IP} \
    --master_port ${MASTER_PORT} \
pretrain.py \
    --trainDataPath           ${TRAIN_DATA_PATH} \
    --evalDataPath            ${EVAL_DATA_PATH} \
    --vocabPath               ${VOCAB_PATH} \
    --logPath                 ${LOG_PATH} \
    --resume_checkpoint       ${RESUME_CHECKPOINT_PATH} \
    --hidden_size             ${HIDDEN_SIZE} \
    --num_hidden_layers       ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads     ${NUM_ATTENTION_HEADS} \
    --intermediate_size       ${INTERMEDIATE_SIZE} \
    --output_dir              ${OUTPUT_DIR} \
    --learning_rate           ${LEARNING_RATE} \
    --evaluation_strategy     ${EVALUATION_STRATEGY} \
    --per_device_train_batch_size   ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size    ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps   ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --weight_decay            ${WEIGHT_DECAY} \
    --max_steps               ${MAX_STEPS} \
    --num_train_epochs        ${NUM_TRAIN_EPOCHS} \
    --lr_scheduler_type       ${LR_SCHEDULER_TYPE} \
    --warmup_steps            ${WARMUP_STEPS} \
    --logging_dir             ${TENSORBOARD_LOG_PATH} \
    --logging_steps           ${LOGGING_STEPS} \
    --save_steps              ${SAVE_STEPS} \
    --save_total_limit        ${SAVE_TOTAL_LIMIT} \
    --eval_steps              ${EVAL_STEPS} \
    --dataloader_num_workers  ${DATALOADER_NUM_WORKERS} \
    --label_smoothing_factor  ${LABEL_SMOOTHING_FACTOR} \
    --ddp_find_unused_parameters  ${DDP_FIND_UNUSED_PARAMETERS} \
    --num_examples_for_epoch  ${NUM_EXAMPLES_FOR_EPOCH}
