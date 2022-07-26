#!/bin/bash
#SBATCH --job-name=dist_launch_base
#SBATCH --partition=a10
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --output=./dist_launch_base-%A-%a.log
#SBATCH --error=./dist_launch_base-%A-%a.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

export MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`srun --nodes=1 --ntasks=1 --nodelist=${MASTER_HOST} python tools/find_unused_port.py`

export nproc_per_node=4
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export PER_DEVICE_EVAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export LOG_BASE=2

export MODEL_SIZE=base
export HIDDEN_SIZE=768
export NUM_HIDDEN_LAYERS=12
export NUM_ATTENTION_HEADS=12
export INTERMEDIATE_SIZE=3072
export NON_CASUAL_LAYERS=2
export CASUAL_LAYERS=0
export SHARED_NON_CASUAL_DECODER=false
#export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-50000

srun ./dist_run.sh
