"""
Author: Da Ma
Date: 2022.01.05
Description: the pre-training model file. Refer to https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert/modeling_bert.py
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertConfig
from transformers import BertModel, BertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainingHeads


class PlmConfig(BertConfig):
    """
    Config of our plm.
    NOTE:
        1. [2022.01.06] Support MLM solely. Different from Bert, the vocabulary size changes.
    """

    def __init__(
            self,
            vocab_size=20000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            **kwargs
        )


@dataclass
class PlmModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # [FIXME: INTRODUCE MORE PLM Model OUTPUTS HERE]


@dataclass
class PlmForPreTrainingOutput:
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None


class PlmPreTrainedModel(BertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PlmConfig
    base_model_prefix = "plm"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _reorder_cache(self, past, beam_idx):
        pass


class PlmModel(PlmPreTrainedModel):
    """
    NOTE:
        1. [2022.01.06]
            Support MLM solely.
    """

    def __init__(self, config: PlmConfig):
        super().__init__(config)

        self.bertEncoder = BertModel(config)

        # [FIXME: INTRODUCE MORE MODULES HERE!]

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """
        Include two parts:
            1. dataflow
            2. output
        """
        # =================== DATAFLOW =================== #
        bertEncoderOutput = self.bertEncoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # [FIXME: INTRODUCE MORE FORWARD DATAFLOW HERE!]

        # =================== OUTPUT =================== #
        # add required output to the [PlmOutput] class above and
        # extract corresponding results from the dataflow.

        # [FIXME: INTRODUCE MORE OUTPUTS HERE!]
        if not return_dict:
            return bertEncoderOutput
        else:
            return PlmModelOutput(
                last_hidden_state=bertEncoderOutput.last_hidden_state,
                pooler_output=bertEncoderOutput.pooler_output,
                hidden_states=bertEncoderOutput.hidden_states,
                attentions=bertEncoderOutput.attentions,
                cross_attentions=bertEncoderOutput.cross_attentions,
            )

    # NOTE: These two following methods are important! They are related
    # to the embedding tying.
    def get_input_embeddings(self):
        return self.bertEncoder.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bertEncoder.embeddings.word_embeddings = value


class PlmForPreTraining(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)

        self.plm = PlmModel(config)

        # ================== SOME OUTPUT LAYERS ================== #
        self.mlm = BertPreTrainingHeads(config)
        self.clm = ...

        # [FIXME: INTRODUCE OUTPUT LAYERS FOR DIFFERENT TASK HERE!]

        self.init_weights()

    # NOTE: These two following methods are important! They are related
    # to the embedding tying.
    def get_output_embeddings(self):
        return self.mlm.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # =================== DATAFLOW =================== #
        plmOutputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        sequence_output, pooled_output = plmOutputs[:2]
        prediction_scores, seq_relationship_score = self.mlm(
            sequence_output, pooled_output
        )

        # [FIXME: INTRODUCE OUTPUT LAYER DATA FLOW HERE!]

        # =================== OUTPUT =================== #
        # [FIXME: INTRODUCE MORE OUTPUT LOGITS HERE!]
        if not return_dict:
            return prediction_scores, seq_relationship_score
        else:
            return PlmForPreTrainingOutput(
                prediction_logits=prediction_scores,
                seq_relationship_logits=seq_relationship_score,
            )


class PlmForSequenceClassification(PlmPreTrainedModel):
    def __init__(self, config: PlmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.plm = PlmModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # =================== DATAFLOW =================== #
        plmOutputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = plmOutputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # ================== LOSS ================== #
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + plmOutputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=plmOutputs.hidden_states,
            attentions=plmOutputs.attentions,
        )
