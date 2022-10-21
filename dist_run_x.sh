#!/bin/bash

####################
###### 必选参数 ######
####################

ddp_required_params=(
  SLURM_NNODES
  SLURM_NODEID
  MASTER_HOST
  MASTER_PORT
  nproc_per_node
)
model_required_params=(
  model_type
  model_size
  log_cross_attn
  hidden_size
  num_hidden_layers
  num_attention_heads
  intermediate_size
  non_casual_layers
  casual_layers
  mapper
  t5_mode
)
train_required_params=(
  per_device_train_batch_size
  per_device_eval_batch_size
  gradient_accumulation_steps
  eval_accumulation_steps
)

####################
###### 参数检测 ######
####################
legal=true

for slot in ${ddp_required_params[@]} ${model_required_params[@]} ${train_required_params[@]}; do
  if [[ -z "${!slot}" ]]; then
    echo "Require: ${slot}"
    legal=false
  fi
done

[[ ${legal} == false ]] && exit 1;

####################
###### DDP参数 ######
####################
echo "<node_rank=${SLURM_NODEID}, nnodes=${SLURM_NNODES}> | MASTER_HOST=${MASTER_HOST}, \
MASTER_PORT=${MASTER_PORT} | model_size=${model_size}, nproc_per_node=${nproc_per_node}"

set -e

########################
###### data path #######
########################
CORPUS_PATH=${CORPUS_PATH:-"${HOME}/Dataset/NLP/TencentMedicalPretrain/corpus"}
TRAIN_DATA_PATH_STAGE1="${CORPUS_PATH}/processed/train_stage1"
TRAIN_DATA_PATH_STAGE2="${CORPUS_PATH}/processed/train_stage2"
EVAL_DATA_PATH="${CORPUS_PATH}/processed/dev"
VOCAB_PATH=${VOCAB_PATH:-"${CORPUS_PATH}/tokenizer/vocab.txt"}

########################
###### train path ######
########################
OUTPUT_DIR=${CORPUS_PATH}/ckpts/${model_type}_${model_size}

# +非自回归层数
if [ ${non_casual_layers} -gt 0 ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_ndec${non_casual_layers}
fi

# +自回归层数
if [ ${casual_layers} -gt 0 ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_dec${casual_layers}
fi

# +log底数
OUTPUT_DIR=${OUTPUT_DIR}_log${mapper:'log(x,2)'}

# +log_cross_attn
if [ ${log_cross_attn} == false ]; then
  OUTPUT_DIR=${OUTPUT_DIR}_mn
else
  OUTPUT_DIR=${OUTPUT_DIR}_mlogn
fi

echo ${OUTPUT_DIR}

########################
###### other path ######
########################
LOG_PATH="${OUTPUT_DIR}" # None means output to the stdout
TENSORBOARD_LOG_PATH="${OUTPUT_DIR}/tensorboard"

########################
######## Start  ########
########################
set -x

OMP_NUM_THREADS=1 \
python -um torch.distributed.launch \
    --nnodes          ${SLURM_NNODES} \
    --node_rank       ${SLURM_NODEID} \
    --nproc_per_node  ${nproc_per_node} \
    --master_addr     ${MASTER_HOST} \
    --master_port     ${MASTER_PORT} \
main_x.py \
    --seed                          ${seed:-42} \
    --fp16                          ${fp16:-false} \
    --gradient_checkpointing        ${gradient_checkpointing:-false} \
    --ddp_find_unused_parameters    ${ddp_find_unused_parameters:-false} \
    --ignore_data_skip              false \
    \
    --trainDataPath                 ${TRAIN_DATA_PATH_STAGE1} \
    --trainDataPath2                ${TRAIN_DATA_PATH_STAGE2} \
    --evalDataPath                  ${EVAL_DATA_PATH} \
    --vocabPath                     ${VOCAB_PATH} \
    --logPath                       ${LOG_PATH} \
    --train_file_pattern            ${train_file_pattern:-'epoch-*.pretrain.data'} \
    --max_length                    ${max_length:-512} \
    --max_length2                   ${max_length2:-1024} \
    --mapper                        ${mapper:-'log(x,2)'} \
    --cls_len_disturb_rate          ${cls_len_disturb_rate:-0} \
    \
    --resume_checkpoint             ${resume_checkpoint:-none} \
    --model_type                    ${model_type} \
    --t5_mode                       ${t5_mode} \
    --log_cross_attn                ${log_cross_attn} \
    --hidden_size                   ${hidden_size} \
    --num_hidden_layers             ${num_hidden_layers} \
    --num_attention_heads           ${num_attention_heads} \
    --intermediate_size             ${intermediate_size} \
    --num_non_casual_decoder_layers ${non_casual_layers} \
    --num_casual_decoder_layers     ${casual_layers} \
    \
    --output_dir                    ${OUTPUT_DIR} \
    --learning_rate                 ${learning_rate:-1e-4} \
    --evaluation_strategy           ${evaluation_strategy:-steps} \
    --per_device_train_batch_size   ${per_device_train_batch_size} \
    --per_device_eval_batch_size    ${per_device_eval_batch_size} \
    --gradient_accumulation_steps   ${gradient_accumulation_steps} \
    --eval_accumulation_steps       ${eval_accumulation_steps} \
    --weight_decay                  ${weight_decay:-0.01} \
    --max_steps                     ${max_steps:-1000000} \
    --num_train_epochs              ${num_train_epochs:-8} \
    --num_train_epochs2             ${num_train_epochs2:-2} \
    --lr_scheduler_type             ${lr_scheduler_type:-linear} \
    --warmup_ratio                  ${warmup_ratio:-0.05} \
    --logging_dir                   ${TENSORBOARD_LOG_PATH} \
    --logging_first_step            true \
    --logging_steps                 ${logging_steps:-1000} \
    \
    --save_steps                    ${save_steps:-50000} \
    --save_total_limit              ${save_total_limit:-10} \
    --eval_steps                    ${eval_steps:-50000} \
    --dataloader_num_workers        ${dataloader_num_workers:-6} \
    --label_smoothing_factor        ${label_smoothing_factor:-0} \
    --optim                         ${optim:-adafactor} \
    --load_best_model_at_end        true \
    --metric_for_best_model         ${metric_for_best_model:-loss} \
    --greater_is_better             ${greater_is_better:-false}
