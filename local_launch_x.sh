#!/bin/bash

set -e

export MASTER_HOST=localhost
export MASTER_PORT=$(python tools/find_unused_port.py)

export SLURM_NNODES=1
export SLURM_NODEID=0

export nproc_per_node=1
export per_device_train_batch_size=8
export per_device_eval_batch_size=8
export gradient_accumulation_steps=4
export eval_accumulation_steps=1

export seed=42
export gradient_checkpointing=false
export ddp_find_unused_parameters=false  # Longformer need find_unused_para to be true
export max_steps=-1
export log_base=2
export cls_len_disturb_rate=0.5

export learning_rate=5e-4

export model_type=loglm  # loglm | longloglm | bart
export model_size=tiny
export log_cross_attn=true
export hidden_size=512
export num_hidden_layers=3
export num_attention_heads=8
export intermediate_size=2048
export non_casual_layers=0
export casual_layers=3

export t5_mode=true

export logging_steps=200
export save_steps=2000
export eval_steps=2000

export resume_checkpoint=$HOME/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/loglm_tiny_dec3_log2_mlogn/checkpoint-12000
export optim=adafactor  # adamw_torch | adafactor

./dist_run_x.sh
