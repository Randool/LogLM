#!/bin/bash
#SBATCH --job-name=dist_launch_tiny
#SBATCH --partition=2080ti,gpu,a10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --output=./dist_launch_tiny-%A-%a.log
#SBATCH --error=./dist_launch_tiny-%A-%a.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]
#SBATCH --array=0

set -e

export MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`srun --nodes=1 --ntasks=1 --nodelist=${MASTER_HOST} python tools/find_unused_port.py`

export nproc_per_node=2
export PER_DEVICE_TRAIN_BATCH_SIZE=64
export PER_DEVICE_EVAL_BATCH_SIZE=64
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export MAX_STEPS=1000000

export LOG_BASE=2

export MODEL_SIZE=tiny
export HIDDEN_SIZE=256
export NUM_HIDDEN_LAYERS=3
export NUM_ATTENTION_HEADS=8
export INTERMEDIATE_SIZE=1024
export NON_CASUAL_LAYERS=0
export CASUAL_LAYERS=3
#export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-50000

srun ./dist_run_x.sh
