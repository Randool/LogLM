#!/bin/bash

set -e

export MASTER_HOST=localhost
export MASTER_PORT=$(python tools/find_unused_port.py)
export SLURM_NNODES=1
export SLURM_NODEID=0
export nproc_per_node=1

export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=2
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export MAX_STEPS=-1

export LOG_BASE=2

export MODEL_TYPE=longloglm
export t5_mode=1
export MODEL_SIZE=tiny
export HIDDEN_SIZE=768
export NUM_HIDDEN_LAYERS=3
export NUM_ATTENTION_HEADS=8
export INTERMEDIATE_SIZE=3072
export NON_CASUAL_LAYERS=0
export CASUAL_LAYERS=3

export log_cross_attn=0
# export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-100000

./dist_run_x.sh
