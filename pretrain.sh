#!/bin/bash
# Author: Da Ma
# Date: 2022.01.07
# Description: launcher of the pre-training script

set -e

# ******************* CONFIGURATION ******************* #
# ==== path ==== #
TRAIN_DATA_PATH="/mnt/xlancefs/home/dm311/workspace/MedicalPretrain/data/pretrain/corpus/small-test-data/processed/train"
EVAL_DATA_PATH="/mnt/xlancefs/home/dm311/workspace/MedicalPretrain/data/pretrain/corpus/small-test-data/processed/train"
VOCAB_PATH="/mnt/xlancefs/home/dm311/workspace/MedicalPretrain/data/tokenizer/vocab.txt"
LOG_PATH=None # None means output to the stdout
# RESUME_CHECKPOINT_PATH="/mnt/xlancefs/home/dm311/workspace/MedicalPretrain/ckpts/MLM/checkpoint-100"
RESUME_CHECKPOINT_PATH=None

# ==== devices ==== #
GPU_CARDS="0,1,2,3,4,5,6,7,8"
NODE_NUM=1
NODE_RANK=0
PROC_NUM_PER_NODE=8
MASTER_IP="127.0.0.1"
MASTER_PORT=8888

# ==== training hyperparameters ==== #
MODEL_SIZE="base" # base | large
TASKS="MLM" # MLM, ...
OUTPUT_DIR="/mnt/xlancefs/home/dm311/workspace/MedicalPretrain/ckpts"
LEARNING_RATE=3e-5
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
EVAL_ACCUMULATION_STEPS=4
WEIGHT_DECAY=0.01
MAX_STEPS=1000000
NUM_TRAIN_EPOCHS=3
LR_SCHEDULER_TYPE="linear"
WARMUP_STEPS=10000
TENSORBOARD_LOG_PATH="/data/home/dama/MedicalPretrain/ckpts/temp"
LOGGING_STEPS=1000
SAVE_STEPS=50000
SAVE_TOTAL_LIMIT=30
EVALUATION_STRATEGY="steps"
EVAL_STEPS=50000
DATALOADER_NUM_WORKERS=4
LABEL_SMOOTHING_FACTOR=0.1
DDP_FIND_UNUSED_PARAMETERS=false

# ******************* PREPARATION ******************* #
OUTPUT_DIR=${OUTPUT_DIR}/${TASKS}

# *************************************************** #

if [[ ${MODEL_SIZE} == 'base' ]]; then
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
CUDA_VISIBLE_DEVICES=${GPU_CARDS} \
python -um torch.distributed.launch \
    --nproc_per_node ${PROC_NUM_PER_NODE} \
    --nnodes ${NODE_NUM} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_IP} \
    --master_port ${MASTER_PORT} \
pretrain.py \
    --trainDataPath ${TRAIN_DATA_PATH} \
    --evalDataPath ${EVAL_DATA_PATH} \
    --vocabPath ${VOCAB_PATH} \
    --logPath ${LOG_PATH} \
    --resume_checkpoint ${RESUME_CHECKPOINT_PATH} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_hidden_layers ${NUM_HIDDEN_LAYERS} \
    --num_attention_heads ${NUM_ATTENTION_HEADS} \
    --intermediate_size ${INTERMEDIATE_SIZE} \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LEARNING_RATE} \
    --evaluation_strategy ${EVALUATION_STRATEGY} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --max_steps ${MAX_STEPS} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_steps ${WARMUP_STEPS} \
    --logging_dir ${TENSORBOARD_LOG_PATH} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --eval_steps ${EVAL_STEPS} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --label_smoothing_factor ${LABEL_SMOOTHING_FACTOR} \
    --ddp_find_unused_parameters ${DDP_FIND_UNUSED_PARAMETERS}
