"""
@File        :  main.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/27
@Version     :  1.0
@Description :  None
"""
import glob
import json
from os.path import join, split
from typing import Optional, List

import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, set_seed

from model import LogLMTokenizer, LogLMConfig, LogLMForPreTraining
from trainer import LogLMTrainer
from utils import get_logger


@dataclass
class DataArguments:
    trainDataPath: str = field(metadata={"help": "training data path"})
    evalDataPath: str = field(metadata={"help": "evaluation data path"})
    num_examples_for_epoch: List[int] = field(metadata={"help": "Training example numbers of each epoch"})
    vocabPath: str = field(metadata={"help": "vocabulary path"})
    logPath: str = field(default=None, metadata={"help": "logger output path"})
    train_file_pattern: str = field(default="epoch-*.pretrain.data.shuf")
    dev_file_pattern: str = field(default="dev.data")

    @property
    def train_data_files(self) -> List[str]:
        train_data_files = glob.glob(join(self.trainDataPath, self.train_file_pattern))
        return sorted(train_data_files, key=lambda x: int(split(x)[-1].split(".")[0][6:]))

    @property
    def eval_data_file(self) -> str:
        return join(self.evalDataPath, self.dev_file_pattern)


@dataclass
class ModelArguments:
    resume_checkpoint: Optional[str] = field(default=None, metadata={"help": "resume from this checkpoint."})
    tasks: Optional[str] = field(default="MLM", metadata={"help": "pre-training tasks"})
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Dimensionality of the encoder layers and the pooler layer"},
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "Number of hidden layers in the Transformer encoder"},
    )
    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder"},
    )
    intermediate_size: Optional[int] = field(
        default=3072,
        metadata={"help": "Dimensionality of the `intermediate` (often named feed-forward) "
                          "layer in the Transformer encoder"},
    )
    hidden_act: Optional[str] = field(
        default="gelu",
        metadata={"help": "The non-linear activation function (function or string) in the encoder and pooler"},
    )
    hidden_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout probability for all fully connected layers in the "
                          "embeddings, encoder, and pooler"},
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities"},
    )
    max_position_embeddings: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum sequence length that this model might ever be used with"},
    )
    type_vocab_size: Optional[int] = field(
        default=2,
        metadata={"help": "The vocabulary size of the `token_type_ids` passed"},
    )
    initializer_range: Optional[float] = field(
        default=0.02,
        metadata={"help": "The standard deviation of the truncated_normal_initializer "
                          "for initializing all weight matrices"},
    )
    layer_norm_eps: Optional[float] = field(
        default=1e-12,
        metadata={"help": "The epsilon used by the layer normalization layers"},
    )
    position_embedding_type: Optional[str] = field(
        default="absolute", metadata={"help": "Type of position embedding"}
    )
    use_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not the model should return the last "
                          "key/values attentions (not used by all models)"},
    )
    shared_non_casual_decoder: Optional[bool] = field(default=False,
                                                      metadata={"help": "Share encoder and non-casual decoder"})
    num_non_casual_decoder_layers: Optional[int] = field(default=3,
                                                         metadata={"help": "Layers number of non-casual decoder"})
    num_casual_decoder_layers: Optional[int] = field(default=3,
                                                     metadata={"help": "Layers number of casual decoder"})


def main(_args: List[str] = None):
    # ******************* Parse Arguments ******************* #
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    trainingArgs, dataArgs, modelArgs = parser.parse_args_into_dataclasses(_args)

    # ******************* Logger ******************* #
    log_path = dataArgs.logPath
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
    logger.info(f"\n====== model parameters ======\n{json.dumps(vars(modelArgs), indent=2)}")
    logger.info(f"\n====== data parameters ======\n{json.dumps(vars(dataArgs), indent=2)}")

    # ******************* Set seed ******************* #
    set_seed(trainingArgs.seed)

    # ******************* Preparation ******************* #
    tokenizer = LogLMTokenizer(dataArgs.vocabPath)
    logger.info(f"[Tokenizer] {len(tokenizer)}")

    config = LogLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=modelArgs.hidden_size,
        num_hidden_layers=modelArgs.num_hidden_layers,
        num_attention_heads=modelArgs.num_attention_heads,
        intermediate_size=modelArgs.intermediate_size,
        hidden_act=modelArgs.hidden_act,
        hidden_dropout_prob=modelArgs.hidden_dropout_prob,
        attention_probs_dropout_prob=modelArgs.attention_probs_dropout_prob,
        max_position_embeddings=modelArgs.max_position_embeddings,
        type_vocab_size=modelArgs.type_vocab_size,
        initializer_range=modelArgs.initializer_range,
        layer_norm_eps=modelArgs.layer_norm_eps,
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type=modelArgs.position_embedding_type,
        use_cache=modelArgs.use_cache,
        shared_non_casual_decoder=modelArgs.shared_non_casual_decoder,
        num_non_casual_decoder_layers=modelArgs.num_non_casual_decoder_layers,
        num_casual_decoder_layers=modelArgs.num_casual_decoder_layers,
    )
    model = LogLMForPreTraining(config)
    logger.info(f"[Model] {sum(map(torch.numel, model.parameters()))} params")

    # ============ Datasets ============ #
    # train datasets configuration list construction
    # assert len(dataArgs.train_data_files) >= trainingArgs.num_train_epochs
    for train_data_file in dataArgs.train_data_files:
        logger.info(f"[Train data]: {train_data_file}")

    # ============ Trainer ============ #
    trainer = LogLMTrainer(
        model=model,
        args=trainingArgs,
        tokenizer=tokenizer,
        train_data_files=dataArgs.train_data_files,
        eval_data_file=dataArgs.eval_data_file,
        num_examples_for_epoch=dataArgs.num_examples_for_epoch,
        logger=logger
    )

    # ******************* Train ******************* #
    logger.info("***** START PRE-TRAINING *****")
    resume_checkpoint = modelArgs.resume_checkpoint
    if resume_checkpoint is None or resume_checkpoint.lower() == "none":
        resume_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("***** FINISHED *****")


if __name__ == "__main__":
    main()
