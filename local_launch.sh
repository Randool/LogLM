#!/bin/bash

set -e

export MASTER_HOST=localhost
export MASTER_PORT=`python tools/find_unused_port.py`
export SLURM_NNODES=1
export SLURM_NODEID=0
export nproc_per_node=8

#export MODEL_SIZE=tiny
#export nproc_per_node=4
#export PER_DEVICE_TRAIN_BATCH_SIZE=128
#export PER_DEVICE_EVAL_BATCH_SIZE=128
#export GRADIENT_ACCUMULATION_STEPS=1
#export EVAL_ACCUMULATION_STEPS=1

export MODEL_SIZE=base
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export PER_DEVICE_EVAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42

# export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-100000

./dist_run.sh
