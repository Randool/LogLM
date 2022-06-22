#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --output=./build_index-%A.log
#SBATCH --error=./build_index-%A.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]

DATA_PATH="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"
INPUT_PATH="${DATA_PATH}/processed/train"

python build_index.py \
    --inputPath ${INPUT_PATH} \
    --processNum 8
