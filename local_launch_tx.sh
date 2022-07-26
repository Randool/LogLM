#!/bin/bash
# Usage: NON_CASUAL_LAYERS=2 CUDA_VISIBLE_DEVICES=0,1,2,3 ./local_launch_tx.sh

set -e

export MASTER_HOST=localhost
export MASTER_PORT=`python tools/find_unused_port.py`
export SLURM_NNODES=1
export SLURM_NODEID=0
export nproc_per_node=4

export PER_DEVICE_TRAIN_BATCH_SIZE=64
export PER_DEVICE_EVAL_BATCH_SIZE=64
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export LOG_BASE=2

export MODEL_SIZE=base
export HIDDEN_SIZE=768
export NUM_HIDDEN_LAYERS=12
export NUM_ATTENTION_HEADS=12
export INTERMEDIATE_SIZE=3072
export NON_CASUAL_LAYERS=${NON_CASUAL_LAYERS}  # Require: NON_CASUAL_LAYERS
export CASUAL_LAYERS=0
export SHARED_NON_CASUAL_DECODER=false

export CORPUS_PATH="/dockerdata/yunyanzhang/dlf/datapath"
export VOCAB_PATH="/dockerdata/yunyanzhang/dlf/LogLM/preprocess_data/tokenizer/vocab.txt"
# export RESUME_CHECKPOINT_PATH=/dockerdata/yunyanzhang/dlf/datapath/ckpts/base/checkpoint-100000

bash dist_run.sh
