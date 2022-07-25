#!/bin/bash
#SBATCH --job-name=dist_launch_tiny
#SBATCH --partition=a10,2080ti,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=10G
#SBATCH --output=./dist_launch-%A.log
#SBATCH --error=./dist_launch-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

export MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`srun --nodes=1 --ntasks=1 --nodelist=${MASTER_HOST} python tools/find_unused_port.py`
echo "NODELIST=${SLURM_NODELIST}, MASTER_HOST: ${MASTER_HOST}, MASTER_PORT: ${MASTER_PORT}"

export nproc_per_node=2
export PER_DEVICE_TRAIN_BATCH_SIZE=64
export PER_DEVICE_EVAL_BATCH_SIZE=64
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42
export LOG_BASE=2

export MODEL_SIZE=tiny
export HIDDEN_SIZE=256
export NUM_HIDDEN_LAYERS=6
export NUM_ATTENTION_HEADS=8
export INTERMEDIATE_SIZE=1024
export NON_CASUAL_LAYERS=0
export CASUAL_LAYERS=0
export SHARED_NON_CASUAL_DECODER=false
#export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-50000

srun ./dist_run.sh
