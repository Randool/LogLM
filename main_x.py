"""
@File        :  main_x.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/8/30
@Version     :  1.0
@Description :  None
"""
import json
from typing import List

import torch
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, set_seed, BartConfig, BartForConditionalGeneration

from main import DataArguments, ModelArguments
from model import LogLMTokenizer, LogLMConfig, LogLMForConditionalGeneration
from trainer_x import Seq2SeqTrainerPro
from utils import get_logger


def main(_args: List[str] = None):
    parser = HfArgumentParser((Seq2SeqTrainingArguments, DataArguments, ModelArguments))
    trainingArgs, data_args, model_args = parser.parse_args_into_dataclasses(_args)
    trainingArgs: Seq2SeqTrainingArguments
    data_args: DataArguments
    model_args: ModelArguments

    # ******************* Logger ******************* #
    log_path = data_args.logPath
    if log_path is None or log_path.lower() == "none":
        log_path = None

    # `torch.distributed.init_process_group` is called in `TrainingArguments`
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    logger = get_logger(__name__, log_path, global_rank, propagate=False)

    for _log_name, _logger in logger.manager.loggerDict.items():
        if _log_name.startswith("transformers.trainer"):
            _logger.addHandler(logger.handlers[0])

    logger.warning(
        f"Global rank: {global_rank}, Process rank: {trainingArgs.local_rank}, device: {trainingArgs.device}, "
        f"n_gpu: {trainingArgs.n_gpu}, distributed training: {bool(trainingArgs.local_rank != -1)}"
    )

    # arguments
    logger.info(f"\n====== Training/evaluation parameters ======\n{trainingArgs.to_json_string()}")
    logger.info(f"\n====== model parameters ======\n{json.dumps(vars(model_args), indent=2)}")
    logger.info(f"\n====== data parameters ======\n{json.dumps(vars(data_args), indent=2)}")

    # ******************* Set seed ******************* #
    set_seed(trainingArgs.seed)

    # ******************* Preparation ******************* #
    tokenizer = LogLMTokenizer.from_pretrained(data_args.vocabPath, log_base=data_args.log_base,
                                               anchored_length=data_args.anchored_length)
    logger.info(f"[Tokenizer] {len(tokenizer)}")

    if "loglm" in model_args.model_type.lower():
        config = LogLMConfig(
            vocab_size=len(tokenizer),
            encoder_layers=model_args.num_hidden_layers,
            encoder_ffn_dim=model_args.hidden_size * 4,
            encoder_attention_heads=16,
            decoder_layers=model_args.num_casual_decoder_layers,
            decoder_ffn_dim=model_args.hidden_size * 4,
            decoder_attention_heads=16,
            d_model=model_args.hidden_size,
            use_longformer=model_args.model_type.lower() == "longloglm",
            log_cross_attn=model_args.log_cross_attn,
        )
        model = LogLMForConditionalGeneration(config)

    elif model_args.model_type.lower() == "bart":
        config = BartConfig(
            vocab_size=len(tokenizer),
            encoder_layers=model_args.num_hidden_layers,
            encoder_ffn_dim=model_args.hidden_size * 4,
            encoder_attention_heads=16,
            decoder_layers=model_args.num_casual_decoder_layers,
            decoder_ffn_dim=model_args.hidden_size * 4,
            decoder_attention_heads=16,
            d_model=model_args.hidden_size,
            decoder_start_token_id=tokenizer.sep_token_id
        )
        model = BartForConditionalGeneration(config)
        tokenizer.set_log_base_or_anchored_length(anchored_length=1)

    else:
        raise ValueError(f"Unrecognized model_type: {model_args.model_type}")

    # print(model)
    logger.info(f"[Model={model_args.model_type}] {sum(map(torch.numel, model.parameters())):.2e} params")

    # ============ Datasets ============ #
    for train_data_file in data_args.train_data_files:
        logger.info(f"[Train data]: {train_data_file}")

    # ============ Trainer ============ #
    trainer = Seq2SeqTrainerPro(
        args=trainingArgs,
        model=model,
        tokenizer=tokenizer,
        train_data_files=data_args.train_data_files,
        eval_data_file=data_args.eval_data_file,
        use_cls_ids=("loglm" in model_args.model_type.lower()),
        t5_mode=model_args.t5_mode,
        logger=logger,
    )

    # ******************* Train ******************* #
    logger.info("***** START PRE-TRAINING *****")
    resume_checkpoint = model_args.resume_checkpoint
    if resume_checkpoint is None or resume_checkpoint.lower() == "none":
        resume_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("***** FINISHED *****")


if __name__ == '__main__':
    main()
