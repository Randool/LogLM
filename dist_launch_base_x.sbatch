#!/bin/bash
#SBATCH --job-name=dist_launch_base
#SBATCH --partition=2080ti,gpu,a10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
#SBATCH --output=./dist_launch_base-%A-%a.log
#SBATCH --error=./dist_launch_base-%A-%a.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,130,144,175]
#SBATCH --array=0

set -e

export MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(srun --nodes=1 --ntasks=1 --nodelist="${MASTER_HOST}" python tools/find_unused_port.py)

export nproc_per_node=4
export per_device_train_batch_size=8
export per_device_eval_batch_size=8
export gradient_accumulation_steps=2
export eval_accumulation_steps=1

export seed=42
export gradient_checkpointing=false
export ddp_find_unused_parameters=false  # Longformer need find_unused_para to be true
export max_steps=-1
export num_train_epochs=6
export num_train_epochs2=2
export mapper='log(x,2)'
export cls_len_disturb_rate=0.5

export learning_rate=5e-4

export model_type=loglm  # loglm | longloglm | bart
export model_size=base
export log_cross_attn=true
export hidden_size=768
export num_hidden_layers=6
export num_attention_heads=12
export intermediate_size=3072
export non_casual_layers=0
export casual_layers=6

export t5_mode=true

export optim=adamw_torch  # adamw_torch | adafactor

srun ./dist_run_x.sh
