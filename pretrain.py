"""
Author: Da Ma
Date: 2022.01.07
Description: pre-training launcher
"""
import json
import logging
import os
from typing import Optional

import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, set_seed

from dataset import JsonLineLargeDataset
from model import LogLMTokenizer, LogLMConfig, LogLMForPreTraining
from trainer import PlmTrainer
from utils import get_logger, is_main_process


@dataclass
class DataArguments:
    trainDataPath: str = field(metadata={"help": "training data path"})
    evalDataPath: str = field(metadata={"help": "evaluation data path"})
    vocabPath: str = field(metadata={"help": "vocabulary path"})
    logPath: str = field(default=None, metadata={"help": "logger output path"})


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


def pretrain():
    # ******************* Parse Arguments ******************* #
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    trainingArgs, dataArgs, modelArgs = parser.parse_args_into_dataclasses()

    # ******************* Logger ******************* #
    logging.basicConfig(level=logging.INFO if is_main_process(trainingArgs.local_rank) else logging.WARN)

    logPath = dataArgs.logPath
    if logPath is None or logPath.lower() == "none":
        logPath = None

    logger = get_logger(__name__, logPath, trainingArgs.local_rank)
    logger.warning(
        f"Process rank: {trainingArgs.local_rank}, device: {trainingArgs.device}, n_gpu: {trainingArgs.n_gpu}, "
        + f"distributed training: {bool(trainingArgs.local_rank != -1)}"
    )

    # arguments
    logger.info(f"\n====== Training/evaluation parameters ======\n{trainingArgs.to_json_string()}")
    logger.info(f"\n====== model parameters ======\n{json.dumps(vars(modelArgs), indent=2)}")
    logger.info(f"\n====== data parameters ======\n{json.dumps(vars(dataArgs), indent=2)}")

    # ******************* Set seed ******************* #
    set_seed(trainingArgs.seed)

    # ******************* Preparation ******************* #
    tokenizer = LogLMTokenizer(dataArgs.vocabPath)
    logger.info(f"Tokenizer: {len(tokenizer)}")

    # ============ Config ============ #
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
        position_embedding_type=modelArgs.position_embedding_type,
        use_cache=modelArgs.use_cache,
        tasks=modelArgs.tasks,
    )

    # ============ Model ============ #
    model = LogLMForPreTraining(config)
    logger.info(f"Model: {sum(map(torch.numel, model.parameters()))} params")

    # ============ Datasets ============ #
    # train datasets configuration list construction
    datasetConfigList = []
    epochDataFiles = []
    index = 0
    for epochFile in sorted(os.listdir(dataArgs.trainDataPath), key=lambda x: int(x.split(".")[0][6:])):
        if not epochFile.startswith("."):
            epochPath = os.path.join(dataArgs.trainDataPath, epochFile)
            epochDataFiles.append(epochPath)
            index += 1

    for index, epochDataFile in enumerate(epochDataFiles):
        datasetConfigList.append({"dataFile": epochDataFile})

    # evaluation dataset
    files = []
    for filename in sorted(os.listdir(dataArgs.evalDataPath)):
        if not filename.startswith("."):
            files.append(os.path.join(dataArgs.evalDataPath, filename))
    evalDataset = JsonLineLargeDataset(files[0], files[1])

    # ============ Trainer ============ #
    trainer = PlmTrainer(
        model=model,
        args=trainingArgs,
        datasetType=JsonLineLargeDataset,
        epochDatasetConfigList=datasetConfigList,
        logPath=logPath,
        eval_dataset=evalDataset,
        tokenizer=tokenizer,
        data_collator=collate_func,
    )

    # ******************* Train ******************* #
    logger.info("***** START PRE-TRAINING *****")
    resume_checkpoint = modelArgs.resume_checkpoint
    if resume_checkpoint.lower() == "none":
        resume_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("***** FINISHED *****")


if __name__ == "__main__":
    pretrain()
