#!/bin/bash
#SBATCH --job-name=run_pipeline.sbatch
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=1-6
#SBATCH --output=./run_pipeline-%A-%a.log
#SBATCH --error=./run_pipeline-%A-%a.log
#SBATCH --exclude=gqxx-01-[027,067,078,118,175]
#SBATCH --array=0,1,2,3,4,5,6,7,8

source activate medical
set -e

date
echo "Array: ${SLURM_ARRAY_TASK_ID}, Host: `hostname`"

CBLUE_ROOT="${HOME}/Dataset/NLP/CBLUEDatasets"

BATCH_SIZE=16
ACC_STEPS=2
LR=3e-5

CKPT_ZOOM="${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts"
MODEL_TYPE="loglm"
MODEL_SIZE="tiny"
MODEL_NAME="checkpoint-1000000"

case ${SLURM_ARRAY_TASK_ID} in
0)
  MODEL_TYPE=bert
  MODEL_ROOT=${HOME}/Dataset/transformers
  MODEL_NAME=chinese-macbert-base
  MODEL_SIZE=base
  CLS_length=1
  DEC=0
  ;;
1)
  MODEL_TYPE=plm
  MODEL_ROOT=${HOME}/remote/plm
  MODEL_NAME=checkpoint-520000
  MODEL_SIZE=base
  CLS_length=1
  DEC=0
  ;;
2)
  MODEL_TYPE=plm
  MODEL_ROOT=${HOME}/remote/plm
  MODEL_NAME=checkpoint-1000000
  MODEL_SIZE=base
  CLS_length=1
  DEC=0
  ;;
3)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_log1024
  MODEL_NAME=checkpoint-500000
  MODEL_SIZE=base
  CLS_length=1
  DEC=0
  ;;
4)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_ndec2_log1024
  MODEL_NAME=checkpoint-500000
  MODEL_SIZE=base
  CLS_length=1
  DEC=2
  ;;
5)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_log2
  MODEL_NAME=checkpoint-1000000
  MODEL_SIZE=base
  CLS_length=-1
  DEC=0
  ;;
6)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_log2
  MODEL_NAME=checkpoint-1000000
  MODEL_SIZE=base
  CLS_length=7
  DEC=0
  ;;
7)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_ndec2_log2
  MODEL_NAME=checkpoint-1000000
  MODEL_SIZE=base
  CLS_length=-1
  DEC=2
  ;;
8)
  MODEL_TYPE=loglm
  MODEL_ROOT=${CKPT_ZOOM}/base_ndec2_log2
  MODEL_NAME=checkpoint-1000000
  MODEL_SIZE=base
  CLS_length=7
  DEC=2
  ;;
esac

SEED=42
FOLD=0
FOLDS=0
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
PWD=`pwd`

for TASK_NAME in CHIP-CTC KUAKE-QIC KUAKE-QTR KUAKE-QQR CHIP-STS; do
  echo "==== ${TASK_NAME} ===="
  cd ${PWD}

  OUTPUT_DIR="${HOME}/tmp/CBLUE_classifier/${TASK_NAME}/${MODEL_TYPE}_${MODEL_SIZE}_${MODEL_NAME}_${SEED}_CLS${CLS_length}"
  if [ ${DEC} -ne 0 ]; then
    OUTPUT_DIR=${OUTPUT_DIR}_DEC${DEC}
  fi

  PYTHONPATH=../.. \
  python main.py \
    --output_dir                  ${OUTPUT_DIR} \
    --report_to                   none \
    --overwrite_output_dir        true \
    \
    --do_train                    true \
    --do_eval                     true \
    --do_predict                  true \
    \
    --dataloader_pin_memory       False \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size  ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --eval_accumulation_steps     500 \
    \
    --learning_rate               ${LR} \
    --weight_decay                2e-2 \
    --max_grad_norm               0.5 \
    --label_smoothing_factor      0 \
    --lr_scheduler_type           cosine \
    \
    --num_train_epochs            8 \
    --warmup_ratio                0.1 \
    --logging_dir                 ${OUTPUT_DIR} \
    \
    --logging_strategy            steps \
    --logging_first_step          true \
    --logging_steps               100 \
    --save_strategy               epoch \
    --evaluation_strategy         epoch \
    \
    --save_total_limit            1 \
    --no_cuda                     false \
    --seed                        ${SEED} \
    --dataloader_num_workers      6 \
    --disable_tqdm                true \
    --load_best_model_at_end      true \
    --metric_for_best_model       f1 \
    --greater_is_better           true \
    \
    --model_type                  ${MODEL_TYPE} \
    --model_path                  ${MODEL_PATH} \
    --layer_wise_lr_decay         0.98 \
    --noise_lambda                0 \
    --anchored_cls_length         ${CLS_length} \
    \
    --cblue_root                  ${CBLUE_ROOT} \
    --task_name                   ${TASK_NAME} \
    --max_length                  128 \
    \
    --use_rdrop                   0
#    --curr_fold                   ${FOLD} \
#    --total_folds                 ${FOLDS}
#    --extra_train_datas           ${HOME}/tmp/CBLUE/${TASK_NAME}/mean_pseudo.json

  rm "${OUTPUT_DIR}"/checkpoint-*/optimizer.pt
done
