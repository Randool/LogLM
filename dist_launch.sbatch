#!/bin/bash
#SBATCH --job-name=dist_launch
#SBATCH --partition=a10
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --output=./dist_launch-%A.log
#SBATCH --error=./dist_launch-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=`srun --nodes=1 --ntasks=1 --nodelist=${MASTER_HOST} python tools/find_unused_port.py`
echo "NODELIST=${SLURM_NODELIST}, MASTER_HOST: ${MASTER_HOST}, MASTER_PORT: ${MASTER_PORT}"

export MASTER_HOST
export MASTER_PORT

#export MODEL_SIZE=tiny
#export nproc_per_node=4
#export PER_DEVICE_TRAIN_BATCH_SIZE=128
#export PER_DEVICE_EVAL_BATCH_SIZE=128
#export GRADIENT_ACCUMULATION_STEPS=1
#export EVAL_ACCUMULATION_STEPS=1

export MODEL_SIZE=base
export nproc_per_node=4
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export PER_DEVICE_EVAL_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export EVAL_ACCUMULATION_STEPS=1

export SEED=42

export SHARED_NON_CASUAL_DECODER=false
#export RESUME_CHECKPOINT_PATH=/mnt/lustre/sjtu/home/lfd98/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base/checkpoint-50000

srun ./dist_run.sh
