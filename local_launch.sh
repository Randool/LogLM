#!/bin/bash

set -e

export MASTER_HOST=localhost
export MASTER_PORT=`python tools/find_unused_port.py`
export SLURM_NNODES=1
export SLURM_NODEID=0
export nproc_per_node=8

export PER_DEVICE_TRAIN_BATCH_SIZE=32
export PER_DEVICE_EVAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export LOG_BASE=3

export MODEL_SIZE=base
export HIDDEN_SIZE=768
export NUM_HIDDEN_LAYERS=12
export NUM_ATTENTION_HEADS=12
export INTERMEDIATE_SIZE=3072
export NON_CASUAL_LAYERS=2
export CASUAL_LAYERS=2
export SHARED_NON_CASUAL_DECODER=false
# export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-100000

./dist_run.sh
