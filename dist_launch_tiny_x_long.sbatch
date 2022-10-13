#!/bin/bash
#SBATCH --job-name=dist_launch_tiny_long
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --output=./dist_launch_tiny_long-%A-%a.log
#SBATCH --error=./dist_launch_tiny_long-%A-%a.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]
#SBATCH --array=0,1

set -e

export MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(srun --nodes=1 --ntasks=1 --nodelist="${MASTER_HOST}" python tools/find_unused_port.py)

export nproc_per_node=2
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

case ${SLURM_ARRAY_TASK_ID} in
0)
  export log_cross_attn=0
  export RESUME_CHECKPOINT_PATH=$HOME/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/longloglm_tiny_dec3_Xnn_log2/checkpoint-600000
  ;;
1)
  export log_cross_attn=1
  export RESUME_CHECKPOINT_PATH=$HOME/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/longloglm_tiny_dec3_log2/checkpoint-600000
  ;;
esac


srun ./dist_run_x.sh
