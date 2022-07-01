#!/bin/bash
#SBATCH --job-name=dist_launch
#SBATCH --partition=a10
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --output=./dist_launch-%A.log
#SBATCH --error=./dist_launch-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

set -e

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=`srun --nodes=1 --ntasks=1 --nodelist=${MASTER_HOST} python tools/find_unused_port.py`
echo "NODELIST=${SLURM_NODELIST}, MASTER_HOST: ${MASTER_HOST}, MASTER_PORT: ${MASTER_PORT}"

export MASTER_HOST
export MASTER_PORT
export MODEL_SIZE=base
export nproc_per_node=4

srun ./dist_run.sh
